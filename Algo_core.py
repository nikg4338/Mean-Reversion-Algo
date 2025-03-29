import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants
START_CASH = 10000
DAYS_IN_YEAR = 365
COOLDOWN_DAYS = 10
MAX_POSITIONS = 5
MAX_HOLDING_DAYS = 15
DAILY_LOSS_LIMIT = -200
TRAILING_STOP_TRIGGER = 0.02
TIME_BASED_TRAILING_DAYS = 10

def backtest_multi_symbol(data, initial_cash=START_CASH):
    data = data.sort_values(["date", "symbol"]).reset_index(drop=True)
    cash = initial_cash
    trades = []
    positions = {}
    cooldown_tracker = {}
    daily_losses = {}
    last_rebalance_date = None

    for _, row in data.iterrows():
        date, sym, price, trade_action = row["date"], row["symbol"], row["close"], row["Trade Action"]
        atr = row.get("ATR", price * 0.02)

        # Adaptive Stop-Loss and Take-Profit using ATR percentiles
        atr_percentile = np.percentile(data["ATR"].dropna(), 75)
        sl_multiplier = 1.5 if atr > atr_percentile else 2.5
        tp_multiplier = 2.0 if atr > atr_percentile else 3.5
        stop_loss_price = price * (1 - (atr * sl_multiplier))
        target_profit = price * (1 + (atr * tp_multiplier))

        # Weekly Rebalancing Check
        if last_rebalance_date and (date - last_rebalance_date).days >= 7:
            positions.clear()  # Sell all positions
            last_rebalance_date = date

        # Manage daily losses
        daily_losses[date] = daily_losses.get(date, 0)
        if sum(daily_losses.values()) <= DAILY_LOSS_LIMIT * 3:
            continue  # Stop trading if recent losses exceed 3-day limit

        # Manage Cooldowns
        if sym in cooldown_tracker and (date - cooldown_tracker[sym]).days < COOLDOWN_DAYS:
            continue

        # Trade Entry
        if trade_action == "BUY" and len(positions) < MAX_POSITIONS:
            risk_amount = cash * 0.005
            shares = min(int(risk_amount / atr), int(cash // price))
            if shares > 0:
                positions[sym] = {
                    "entry_date": date,
                    "entry_price": price,
                    "stop_loss_price": stop_loss_price,
                    "target_profit": target_profit,
                    "shares": shares
                }
                cash -= shares * price

        # Trade Exits
        if sym in positions:
            pos = positions[sym]
            holding_days = (date - pos["entry_date"]).days
            if price <= pos["stop_loss_price"] or holding_days >= MAX_HOLDING_DAYS:
                reason = "STOP-LOSS" if price <= pos["stop_loss_price"] else "TIME EXIT"
                profit = (price - pos["entry_price"]) * pos["shares"]
                trades.append({
                    "symbol": sym,
                    "entry_date": pos["entry_date"],
                    "exit_date": date,
                    "entry_price": pos["entry_price"],
                    "exit_price": price,
                    "profit": profit,
                    "days_held": holding_days,
                    "exit_reason": reason
                })
                cash += pos["shares"] * price
                cooldown_tracker[sym] = date
                del positions[sym]

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv("final_backtest_results.csv", index=False)
    print("✅ Backtest completed. Results saved to 'final_backtest_results.csv'")
    return trades_df, cash
