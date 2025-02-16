#%%
import pandas as pd
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
#%%
# Parameters
order_value = 9
volume_threshold = 150
min_pips = 15  # Minimum number of pips required for a valid signal (adjust as needed)
pip_value = 0.0001  # Definition of 1 pip for EURUSD

# Load and prepare data
file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled_EURUSD_1min.csv"
data = pd.read_csv(file_path)

# Convert Date and Time to datetime
data['Date'] = data['Date'].astype(str)
data['Time'] = data['Time'].astype(str)
data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

# Convert Close column to float
data['Close'] = data['Close'].astype(float)

# Find local extrema
local_max_indices = signal.argrelextrema(data['Close'].values, np.greater, order=order_value)[0]
local_min_indices = signal.argrelextrema(data['Close'].values, np.less, order=order_value)[0]

# Initialize signal columns
data["buy_signal"] = 0
data["sell_signal"] = 0
data["Close_Position"] = 0

# Apply initial volume filter and assign signals
data.loc[(data.index.isin(local_max_indices)) & (data["Volume"] > volume_threshold), "sell_signal"] = 1
data.loc[(data.index.isin(local_min_indices)) & (data["Volume"] > volume_threshold), "buy_signal"] = 1

# Get indices where a trade signal occurs
trade_indices = data.index[(data["buy_signal"] == 1) | (data["sell_signal"] == 1)].tolist()

# Function to check if trade meets minimum pip movement
def check_pip_movement(start_idx, end_idx, signal_type):
    start_price = data.iloc[start_idx]["Close"]
    price_range = data.iloc[start_idx:end_idx+1]["Close"]
    
    if signal_type == "buy":
        max_movement = price_range.max() - start_price
        return max_movement >= (min_pips * pip_value)
    else:  # sell signal
        max_movement = start_price - price_range.min()
        return max_movement >= (min_pips * pip_value)

# Filter signals and mark close positions
valid_trades = []
for i in range(len(trade_indices) - 1):
    current_idx = trade_indices[i]
    next_idx = trade_indices[i + 1]
    
    # Determine signal type
    if data.iloc[current_idx]["buy_signal"] == 1:
        signal_type = "buy"
    else:
        signal_type = "sell"
    
    # Check if trade meets minimum pip movement
    if check_pip_movement(current_idx, next_idx, signal_type):
        valid_trades.append(current_idx)
        if next_idx - 1 > current_idx:
            data.at[next_idx - 1, "Close_Position"] = 1
    else:
        # Remove invalid signals
        data.at[current_idx, "buy_signal"] = 0
        data.at[current_idx, "sell_signal"] = 0

# Plot results
plt.figure(figsize=(12, 6))
subset = data[-3000:]  # Last 3000 rows

plt.plot(subset["datetime"], subset["Close"], label="Close Price", color="blue")

# Plot Buy signals
buy_signals = subset[subset["buy_signal"] == 1]
plt.scatter(buy_signals["datetime"], buy_signals["Close"], marker="^", color="green", label="Buy Signal", alpha=1)

# Plot Sell signals
sell_signals = subset[subset["sell_signal"] == 1]
plt.scatter(sell_signals["datetime"], sell_signals["Close"], marker="v", color="red", label="Sell Signal", alpha=1)

# Plot Close Position indicators
close_positions = subset[subset["Close_Position"] == 1]
plt.scatter(close_positions["datetime"], close_positions["Close"], marker="x", color="black", label="Close Position", alpha=1)

plt.xlabel("Datetime")
plt.ylabel("Close Price")
plt.title(f"Trade Signals and Close Positions (Min {min_pips} Pips Movement)")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Show the plot
plt.show()
#%%
# Save the updated DataFrame
data.to_csv("trade_signals.csv", index=False)

# %%
# Count buy and sell signals
num_buy_signals = data[data["buy_signal"] == 1].shape[0]
num_sell_signals = data[data["sell_signal"] == 1].shape[0]

# Total number of data points
total_data_points = data.shape[0]

# Compute ratios
buy_ratio = num_buy_signals / total_data_points
sell_ratio = num_sell_signals / total_data_points

# Print results
print(f"Number of Buy Signals: {num_buy_signals}")
print(f"Number of Sell Signals: {num_sell_signals}")
print(f"Buy Signal Ratio: {buy_ratio:.4f}")
print(f"Sell Signal Ratio: {sell_ratio:.4f}")
# %%
