
#%%
import pandas as pd
from backtesting import Backtest, Strategy

# Load dataset
file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv"
data = pd.read_csv(file_path, header=0)

# 🔍 Debug: Check if "signal" column contains strings
print("Unique values in 'signal':", data['Signal'].unique())

# ✅ Convert "Signal" column from string to integer values
data["Signal"] = data["Signal"].map({"Buy": 1, "Sell": -1}).fillna(0).astype(int)

# ✅ Convert "Close_Position" column to integer
data["Close_Position"] = data["Close_Position"].fillna(0).astype(int)

# Function to extract signals and close positions
def SIGNAL():
    return data['Signal']

def CLOSE_POS():
    return data['Close_Position']

# Define your strategy class
class MyStrategy(Strategy):
    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)
        self.close_position = self.I(CLOSE_POS)

    def next(self):
        super().next()

        # Close open position if close_position is triggered
        if self.close_position == 1:
            if self.position:
                self.position.close()

        # Open new position based on signal
        if self.signal1 == 1:
            if not self.position:
                self.buy()
            elif self.position.size < 0:
                self.position.close()
                self.buy()

        elif self.signal1 == -1:
            if not self.position:
                self.sell()
            elif self.position.size > 0:
                self.position.close()
                self.sell()

# ✅ Run the backtest
bt = Backtest(data, MyStrategy, cash=1000)
stats = bt.run()
print(stats)

# %%
import pandas as pd
import numpy as np

# Load dataset
file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv"
data = pd.read_csv(file_path)

# Convert columns to string
data['Date'] = data['Date'].astype(str)  
data['Time'] = data['Time'].astype(str)
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

# Convert numerical columns
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
data[numeric_cols] = data[numeric_cols].astype(float)

# Convert signals to numeric values
data["Signal"] = data["Signal"].map({"Buy": 1, "Sell": -1}).fillna(0).astype(int)
data["Close_Position"] = data["Close_Position"].fillna(0).astype(int)

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
    if row["Close_Position"] == 1 and position is not None:
        if position == "Long":
            profit = (row["Close"] - entry_price) * 10000  # Pip profit
        elif position == "Short":
            profit = (entry_price - row["Close"]) * 10000

        balance += profit  # Update balance
        trade_log.append({"Datetime": row["datetime"], "Action": "Close", "Price": row["Close"], "Balance": balance})
        position = None  # Close position

    # Open new position
    if row["Signal"] == 1 and position is None:
        position = "Long"
        entry_price = row["Close"]
        trade_log.append({"Datetime": row["datetime"], "Action": "Buy", "Price": entry_price, "Balance": balance})

    elif row["Signal"] == -1 and position is None:
        position = "Short"
        entry_price = row["Close"]
        trade_log.append({"Datetime": row["datetime"], "Action": "Sell", "Price": entry_price, "Balance": balance})

# Convert trade log to DataFrame
trade_results = pd.DataFrame(trade_log)
print(trade_results)

# %%
import matplotlib.pyplot as plt

# Convert trade log to DataFrame
trade_results["Datetime"] = pd.to_datetime(trade_results["Datetime"])

# Plot balance over time (Profit/Loss Curve)
plt.figure(figsize=(12, 6))
plt.plot(trade_results["Datetime"], trade_results["Balance"], label="Balance", color="blue")
plt.xlabel("Datetime")
plt.ylabel("Account Balance (USD)")
plt.title("Profit/Loss Curve")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# Plot trades on price chart
plt.figure(figsize=(12, 6))
plt.plot(data["Datetime"], data["Close"], label="Close Price", color="black", alpha=0.6)

# Buy signals
buy_trades = trade_results[trade_results["Action"] == "Buy"]
plt.scatter(buy_trades["Datetime"], buy_trades["Price"], marker="^", color="green", label="Buy", alpha=1, s=100)

# Sell signals
sell_trades = trade_results[trade_results["Action"] == "Sell"]
plt.scatter(sell_trades["Datetime"], sell_trades["Price"], marker="v", color="red", label="Sell", alpha=1, s=100)

# Close position markers
close_trades = trade_results[trade_results["Action"] == "Close"]
plt.scatter(close_trades["Datetime"], close_trades["Price"], marker="x", color="blue", label="Close", alpha=1, s=100)

plt.xlabel("Datetime")
plt.ylabel("Close Price")
plt.title("Trade Entry/Exit Points")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

# %%
