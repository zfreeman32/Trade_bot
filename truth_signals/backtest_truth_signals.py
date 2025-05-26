
#%% Import necessary libraries

"""
🔹 SCRIPT: backtesting_with_backtesting_py.py
🔹 PURPOSE: This script performs a backtest using the `Backtesting.py` library.
🔹 KEY FEATURES:
    - Uses `Backtesting.py`, which automatically manages trades, execution rules, 
       and trade handling logic.
    - Handles position sizing dynamically.
    - Trade execution may be delayed based on `Backtesting.py`'s internal logic.
    - Includes built-in trade features like commissions, slippage, and realistic fills.

"""

import pandas as pd
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_30min_sampled_signals.csv"
data = pd.read_csv(file_path, header=0)

#  Convert date and time columns into a single datetime column
data["datetime"] = pd.to_datetime(data["datetime"])  # Already in dataset

#  Ensure signals and close positions are integer values
data["long_signal"] = data["long_signal"].fillna(0).astype(int)
data["short_signal"] = data["short_signal"].fillna(0).astype(int)
data["close_position"] = data["close_position"].fillna(0).astype(int)

#  Function to extract signals
def LONG_SIGNAL():
    return data['long_signal']

def SHORT_SIGNAL():
    return data['short_signal']

def CLOSE_POS():
    return data['close_position']

#  Define strategy
class MyStrategy(Strategy):
    def init(self):
        super().init()
        self.long_signal = self.I(LONG_SIGNAL)
        self.short_signal = self.I(SHORT_SIGNAL)
        self.close_position = self.I(CLOSE_POS)

    def next(self):
        super().next()

        # Get current index in the backtest data
        idx = self.data.index[-1]

        # Close open position if close_position is triggered
        if self.close_position[idx] == 1:
            if self.position:
                self.position.close()

        # Handle long signals
        if self.long_signal[idx] == 1:
            if not self.position:
                self.buy()
            elif self.position.size < 0:  # If in short position, close it and go long
                self.position.close()
                self.buy()

        # Handle short signals
        elif self.short_signal[idx] == 1:
            if not self.position:
                self.sell()
            elif self.position.size > 0:  # If in long position, close it and go short
                self.position.close()
                self.sell()

#  Run backtest
bt = Backtest(data, MyStrategy, cash=1000)
stats = bt.run()
print(stats)

# Extract trade log from backtest results
# Ensure trades DataFrame exists
trades = stats._trades.copy() if stats._trades is not None else pd.DataFrame(columns=["EntryBar", "ExitBar", "EntryPrice", "ExitPrice", "PnL"])

# Ensure EntryBar and ExitBar exist before mapping
if "EntryBar" in trades.columns and "ExitBar" in trades.columns:
    trades["EntryTime"] = data["datetime"].iloc[trades["EntryBar"].astype(int)].values
    trades["ExitTime"] = data["datetime"].iloc[trades["ExitBar"].astype(int)].values

else:
    trades = pd.DataFrame(columns=["EntryTime", "ExitTime", "EntryPrice", "ExitPrice", "PnL"])

# Extract equity curve (Balance over time)
equity_curve = stats._equity_curve
equity_curve["Time"] = pd.to_datetime(data["datetime"][:len(equity_curve)])  # Match timestamps

# Plot Profit/Loss Curve (Account Balance Over Time)
plt.figure(figsize=(12, 6))
plt.plot(equity_curve["Time"], equity_curve["Equity"], label="Balance", color="blue")
plt.xlabel("Datetime")
plt.ylabel("Account Balance (USD)")
plt.title("Profit/Loss Curve - Backtesting.py")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot Trade Entry/Exit Points on Price Chart
plt.figure(figsize=(12, 6))
plt.plot(data["datetime"], data["Close"], label="Close Price", color="black", alpha=0.6)

# Buy signals
buy_trades = trades[trades["Size"] > 0]
plt.scatter(buy_trades["EntryTime"], buy_trades["EntryPrice"], marker="^", color="green", label="Buy", alpha=1, s=100)

# Sell signals
sell_trades = trades[trades["Size"] < 0]
plt.scatter(sell_trades["EntryTime"], sell_trades["EntryPrice"], marker="v", color="red", label="Sell", alpha=1, s=100)

# Close position markers
plt.scatter(trades["ExitTime"], trades["ExitPrice"], marker="x", color="blue", label="Close", alpha=1, s=100)

plt.xlabel("Datetime")
plt.ylabel("Close Price")
plt.title("Trade Entry/Exit Points - Backtesting.py")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()
# %%
"""
🔹 SCRIPT: manual_backtesting_vectorized.py
🔹 PURPOSE: This script performs a manual, vectorized backtest without using 
            `Backtesting.py`, allowing full control over trade execution.
🔹 KEY FEATURES:
    - Manages trade execution using a **manual loop**, executing trades immediately 
       when signals appear.
    - Uses **fixed position sizing** (assumes 1,000 units per pip movement).
    - Does not include commissions or slippage unless explicitly added.
    - Trades are closed immediately when `close_position == 1`, without waiting 
       for confirmation on the next bar.
    - Keeps a trade log and calculates profit/loss manually.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_30min_sampled_signals.csv"
data = pd.read_csv(file_path)

# Convert columns to string
data['Date'] = data['Date'].astype(str)  
data['Time'] = data['Time'].astype(str)
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

# Convert numerical columns
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
data[numeric_cols] = data[numeric_cols].astype(float)

# Convert signals to numeric values
data["long_signal"] = data["long_signal"].astype(int)
data["short_signal"] = data["short_signal"].astype(int)
data["close_position"] = data["close_position"].fillna(0).astype(int)

# Initialize backtesting variables
initial_balance = 1000
balance = initial_balance
position = None  # 'Long' or 'Short'
entry_price = None
trade_log = []

# Vectorized processing
for i in range(len(data)):
    row = data.iloc[i]

    # Close existing position if needed
    if row["close_position"] == 1 and position is not None:
        if position == "Long":
            profit = (row["Close"] - entry_price) * 1000  # Pip profit
        elif position == "Short":
            profit = (entry_price - row["Close"]) * 1000

        balance += profit  # Update balance
        trade_log.append({"datetime": row["datetime"], "Action": "Close", "Price": row["Close"], "Balance": balance})
        position = None  # Close position

    # Open new position
    if row["long_signal"] == 1 and position is None:
        position = "Long"
        entry_price = row["Close"]
        trade_log.append({"datetime": row["datetime"], "Action": "Buy", "Price": entry_price, "Balance": balance})

    elif row["short_signal"] == 1 and position is None:
        position = "Short"
        entry_price = row["Close"]
        trade_log.append({"datetime": row["datetime"], "Action": "Sell", "Price": entry_price, "Balance": balance})

# Convert trade log to DataFrame
trade_results = pd.DataFrame(trade_log)
print(trade_results)

# Convert trade log to DataFrame
trade_results["datetime"] = pd.to_datetime(trade_results["datetime"])

# Plot balance over time (Profit/Loss Curve)
plt.figure(figsize=(12, 6))
plt.plot(trade_results["datetime"], trade_results["Balance"], label="Balance", color="blue")
plt.xlabel("datetime")
plt.ylabel("Account Balance (USD)")
plt.title("Profit/Loss Curve")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot trades on price chart
plt.figure(figsize=(12, 6))
plt.plot(data["datetime"], data["Close"], label="Close Price", color="black", alpha=0.6)

# Buy signals
buy_trades = trade_results[trade_results["Action"] == "Buy"]
plt.scatter(buy_trades["datetime"], buy_trades["Price"], marker="^", color="green", label="Buy", alpha=1, s=100)

# Sell signals
sell_trades = trade_results[trade_results["Action"] == "Sell"]
plt.scatter(sell_trades["datetime"], sell_trades["Price"], marker="v", color="red", label="Sell", alpha=1, s=100)

# Close position markers
close_trades = trade_results[trade_results["Action"] == "Close"]
plt.scatter(close_trades["datetime"], close_trades["Price"], marker="x", color="blue", label="Close", alpha=1, s=100)

plt.xlabel("datetime")
plt.ylabel("Close Price")
plt.title("Trade Entry/Exit Points")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %%
