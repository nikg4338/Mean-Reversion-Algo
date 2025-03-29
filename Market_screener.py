import pandas as pd
import numpy as np
from yahooquery import Screener, Ticker
import joblib
import warnings
from time import sleep

warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-6)
    return 100 - (100 / (1 + rs))

def compute_z_score(series, window=30):
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std(ddof=0)
    return (series - mean) / (std + 1e-6)

def compute_sharpe(df, window=10):
    df["Return"] = df["close"].pct_change()
    df["Rolling_Mean"] = df["Return"].rolling(window=window).mean()
    df["Rolling_Std"] = df["Return"].rolling(window=window).std()
    df["Sharpe_Ratio"] = df["Rolling_Mean"] / (df["Rolling_Std"] + 1e-6)
    df.drop(columns=["Rolling_Mean", "Rolling_Std"], inplace=True)
    return df

def compute_pairwise_z(df):
    sector_mean = df.groupby("sector")["close"].transform("mean")
    sector_std = df.groupby("sector")["close"].transform("std")
    df["Pairwise_Z"] = (df["close"] - sector_mean) / (sector_std + 1e-6)
    return df

def compute_downside_volatility(df, window=20):
    df["Negative_Returns"] = df["Return"].where(df["Return"] < 0, 0)
    df["Downside_Volatility"] = df["Negative_Returns"].rolling(window=window).std()
    df["Downside_Volatility"].fillna(0, inplace=True)
    return df

def fetch_historical_data(tickers, start, end, batch_size=5):
    all_data = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for attempt in range(2):
            try:
                print(f"📦 Fetching batch {i // batch_size + 1}, attempt {attempt + 1}")
                batch_data = Ticker(batch).history(start=start, end=end).reset_index()
                if not batch_data.empty:
                    all_data.append(batch_data)
                break
            except Exception as e:
                print(f"⚠️ Error fetching batch {i // batch_size + 1}: {e}")
                sleep(2)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def extract_fundamentals(ticker, financial_data, summary_detail, asset_profiles, earnings_data):
    f = financial_data.get(ticker, {})
    s = summary_detail.get(ticker, {})
    a = asset_profiles.get(ticker, {})
    e = earnings_data.get(ticker, {})

    if not isinstance(f, dict): f = {}
    if not isinstance(s, dict): s = {}
    if not isinstance(a, dict): a = {}
    if not isinstance(e, dict): e = {}

    # Sector
    sector = a.get("sector", "Unknown")

    # Earnings surprise
    try:
        quarterly = e.get("earningsChart", {}).get("quarterly", [])
        earnings_surprise = quarterly[-1].get("surprisePercent", np.nan) if quarterly else np.nan
    except Exception:
        earnings_surprise = np.nan

    return {
        "sector": sector,
        "debt_to_equity": f.get("debtToEquity", np.nan),
        "operating_margin": f.get("operatingMargins", np.nan),
        "net_margin": f.get("netMargins", np.nan),
        "market_cap": s.get("marketCap", np.nan),
        "earnings_surprise": earnings_surprise
    }

def run_screener():
    model = joblib.load("ml_model_trained.pkl")

    screener_names = [
        'most_actives',
        'undervalued_large_caps',
        'undervalued_growth_stocks',
        'day_gainers',
        'top_mutual_funds'
    ]

    print("\n🔍 Fetching tickers from screeners...")
    s = Screener()
    all_tickers = set()

    for screener in screener_names:
        try:
            data = s.get_screeners(screener, count=25)
            quotes = data.get(screener, {}).get("quotes", [])
            tickers = [q["symbol"] for q in quotes if "symbol" in q]
            all_tickers.update(tickers)
            print(f"✅ {len(tickers)} tickers from '{screener}'")
        except Exception as e:
            print(f"❌ Error loading screener '{screener}': {e}")

    all_tickers = sorted(all_tickers)
    if not all_tickers:
        print("❌ No tickers found. Exiting.")
        return

    print("\n📊 Fetching historical price data...")
    df = fetch_historical_data(all_tickers, start="2023-10-01", end="2023-12-31")
    if df.empty:
        print("❌ No price data fetched.")
        return

    df = df.groupby("symbol").filter(lambda x: len(x) >= 30)

    print("\n📑 Fetching fundamentals...")
    ticker_obj = Ticker(all_tickers)
    fin = ticker_obj.financial_data
    summ = ticker_obj.summary_detail
    prof = ticker_obj.asset_profile
    earn = ticker_obj.earnings

    print("📌 Enriching dataset with fundamentals...")
    enriched = {t: extract_fundamentals(t, fin, summ, prof, earn) for t in all_tickers}
    for t, row in enriched.items():
        for k, v in row.items():
            df.loc[df["symbol"] == t, k] = v

    # Fill NAs for filters
    df["debt_to_equity"].fillna(999, inplace=True)
    df["operating_margin"].fillna(-1.0, inplace=True)
    df["net_margin"].fillna(-1.0, inplace=True)
    df["market_cap"].fillna(0, inplace=True)

    print("\n🔍 Applying filters...")
    df = df[
        (df["debt_to_equity"] <= 10.0) &
        (df["operating_margin"] >= -2.0) &
        (df["net_margin"] >= -2.0) &
        (df["market_cap"] >= 1e7)
    ]
    if df.empty:
        print("⚠️ All data filtered out.")
        return

    print("\n📐 Calculating indicators...")
    df["RSI"] = df.groupby("symbol")["close"].transform(compute_rsi).fillna(method="bfill").fillna(method="ffill")
    df["Z_Score"] = df.groupby("symbol")["close"].transform(compute_z_score).fillna(0)
    df = compute_sharpe(df)
    df["Sharpe_Ratio"].fillna(0, inplace=True)
    df = compute_pairwise_z(df)
    df["Pairwise_Z"].fillna(0, inplace=True)
    df = compute_downside_volatility(df)
    df["Downside_Volatility"].fillna(0, inplace=True)

    print("\n🤖 Making predictions...")
    features = ["RSI", "Z_Score", "Sharpe_Ratio", "Pairwise_Z"]
    X = df[features]
    df["Predicted"] = model.predict(X)
    df["Confidence"] = model.predict_proba(X).max(axis=1)
    df["Trade Action"] = df["Predicted"].map({0: "HOLD", 1: "BUY", 2: "SELL"})
    df["Pseudo_Chow"] = df["Confidence"] / (df["Downside_Volatility"] + 1e-6)

    print("\n💾 Saving results to Excel...")
    with pd.ExcelWriter("updated_historical_stock_data_with_predictions.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Market Screener Results", index=False)

    print("✅ Done. File saved: updated_historical_stock_data_with_predictions.xlsx")
if __name__ == "__main__":
    run_screener()

