import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def main():
    input_file = "ml_training_results.xlsx"
    output_file = "final_backtest_results.xlsx"

    print(f"\n🔍 Loading historical data for backtesting from '{input_file}'...")
    
    # Load dataset
    try:
        df = pd.read_excel(input_file, sheet_name="ML Predictions")
    except Exception as e:
        print(f"❌ Error loading data from '{input_file}': {e}")
        return

    # Ensure required columns are present
    required_cols = {"symbol", "date", "close", "RSI", "Z_Score", "Trade Action", "Confidence Score", "Sharpe_Ratio"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by="date", inplace=True)

    # Backtest period
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2023, 12, 31)
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    if df.empty:
        print("⚠️ No data for backtest.")
        return

    print("🔍 Sample of trade data for backtesting:")
    print(df.head(5))

    # Initialize Portfolio & Backtest Parameters
    initial_cash = 10_000
    cash = initial_cash
    portfolio = {}
    trades = []
    recent_trades = {}

    max_positions = 10
    SL_MULTIPLIER = 3.5  
    TP_MULTIPLIER = 2.0  
    MAX_HOLD_DAYS = 30  
    COOLDOWN_DAYS = 7  
    TRAILING_STOP_TRIGGER = 0.02  
    CONFIDENCE_THRESHOLD = 0.7  
    SHARPE_THRESHOLD = 0.2  

    # Iterate through trade signals
    for _, row in df.iterrows():
        sym, action, price, confidence, sharpe, trade_date = (
            row["symbol"], row["Trade Action"], row["close"],
            row["Confidence Score"], row["Sharpe_Ratio"], row["date"]
        )

        # Skip if confidence or Sharpe Ratio is too low
        if confidence < CONFIDENCE_THRESHOLD or sharpe < SHARPE_THRESHOLD:
            continue

        # Trade Execution
        if action == "BUY" and len(portfolio) < max_positions and cash >= price:
            allocation = min(cash * 0.15, 2000)
            shares_to_buy = int(allocation // price)
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                cash -= cost
                portfolio.setdefault(sym, []).append({
                    "entry_price": price,
                    "shares": shares_to_buy,
                    "entry_date": trade_date,
                    "tp_price": price * (1 + TP_MULTIPLIER),
                    "sl_price": price * (1 - SL_MULTIPLIER)
                })
                print(f"✅ BUY {shares_to_buy} shares of {sym} at {price}")

    # Save backtest results
    results_df = pd.DataFrame(trades)
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
            results_df.to_excel(writer, sheet_name="Final Backtest Results", index=False)
        print(f"\n✅ Backtest completed. Results saved to '{output_file}' [Sheet: 'Final Backtest Results'].")
    except Exception as e:
        print(f"❌ Error saving backtest results to '{output_file}': {e}")

if __name__ == "__main__":
    main()
