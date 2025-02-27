# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load and prepare data
# ---------------------------

# File Path should be OHLCV Data
df = pd.read_csv(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled.csv')

# Combine Date and Time into a datetime column and sort chronologically.
df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
df = df.sort_values('datetime').reset_index(drop=True)

# ---------------------------
# 2. Compute ATR and Volume Moving Average
# ---------------------------
def compute_atr(df, period=14):
    """
    Compute the Average True Range (ATR) over a given period.
    True Range = max(High-Low, abs(High - Prev_Close), abs(Low - Prev_Close))
    ATR is the rolling mean of True Range.
    """
    df = df.copy()
    df['prev_close'] = df['Close'].shift(1)
    df['tr'] = df.apply(lambda row: max(row['High'] - row['Low'],
                                          abs(row['High'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0,
                                          abs(row['Low'] - row['prev_close']) if pd.notnull(row['prev_close']) else 0),
                        axis=1)
    atr = df['tr'].rolling(window=period, min_periods=1).mean()
    return atr

# Calculate ATR and volume moving average.
df['ATR'] = compute_atr(df, period=14)
df['vol_ma'] = df['Volume'].rolling(window=20, min_periods=1).mean()

# Convert key series to numpy arrays (for speed in our dynamic programming routines).
prices = df['Close'].values
atr = df['ATR'].values
vol_ma = df['vol_ma'].values
volumes = df['Volume'].values

# ---------------------------
# 3. Dynamic Programming for Trade Signals
# ---------------------------
def find_long_trades(prices, atr, volumes, vol_ma,
                     min_profit_threshold=0.002,
                     atr_multiplier=1.0,
                     volume_multiplier=1.0):
    """
    Finds optimal long trades (buy then sell) using dynamic programming.
    Returns a list of tuples (entry_index, exit_index).
    """
    n = len(prices)
    dp = [0.0] * (n + 1)          # Max profit achievable from index i to end.
    trade_decision = [None] * n   # Best trade (entry, exit) starting at index i.
    
    # Iterate backwards
    for i in range(n - 1, -1, -1):
        best_profit = dp[i + 1]   # Option: do nothing.
        best_trade = None
        
        # Only consider if volume is above average.
        if volumes[i] < volume_multiplier * vol_ma[i]:
            dp[i] = best_profit
            trade_decision[i] = None
            continue
        
        # Test potential sell points j > i.
        for j in range(i + 1, n):
            profit_trade = prices[j] - prices[i]
            # Must exceed the basic profit threshold.
            if profit_trade < min_profit_threshold:
                continue
            # Also, ensure the move is strong relative to ATR.
            if profit_trade < atr_multiplier * atr[i]:
                continue
            
            total_profit = profit_trade + (dp[j + 1] if (j + 1) < len(dp) else 0)
            if total_profit > best_profit:
                best_profit = total_profit
                best_trade = (i, j)
        dp[i] = best_profit
        trade_decision[i] = best_trade
    
    # Backtrack to extract nonoverlapping trades.
    trades = []
    i = 0
    while i < n:
        if trade_decision[i] is not None:
            trade = trade_decision[i]
            trades.append(trade)
            i = trade[1] + 1  # Skip to index after exit.
        else:
            i += 1
    return trades

def find_short_trades(prices, atr, volumes, vol_ma,
                      min_profit_threshold=0.008,
                      atr_multiplier=1.0,
                      volume_multiplier=1.0):
    """
    Finds optimal short trades (sell then buy to cover) using dynamic programming.
    Returns a list of tuples (entry_index, exit_index).
    """
    n = len(prices)
    dp = [0.0] * (n + 1)
    trade_decision = [None] * n
    
    for i in range(n - 1, -1, -1):
        best_profit = dp[i + 1]
        best_trade = None
        
        if volumes[i] < volume_multiplier * vol_ma[i]:
            dp[i] = best_profit
            trade_decision[i] = None
            continue
        
        # For short trades, profit is realized when prices drop.
        for j in range(i + 1, n):
            profit_trade = prices[i] - prices[j]
            if profit_trade < min_profit_threshold:
                continue
            if profit_trade < atr_multiplier * atr[i]:
                continue
            
            total_profit = profit_trade + (dp[j + 1] if (j + 1) < len(dp) else 0)
            if total_profit > best_profit:
                best_profit = total_profit
                best_trade = (i, j)
        dp[i] = best_profit
        trade_decision[i] = best_trade
    
    trades = []
    i = 0
    while i < n:
        if trade_decision[i] is not None:
            trade = trade_decision[i]
            trades.append(trade)
            i = trade[1] + 1
        else:
            i += 1
    return trades

#%%
# Get long and short trades.
long_trades = find_long_trades(prices, atr, volumes, vol_ma,
                               min_profit_threshold=0.0015,
                               atr_multiplier=1.0,
                               volume_multiplier=1.0)
short_trades = find_short_trades(prices, atr, volumes, vol_ma,
                                 min_profit_threshold=0.0015,
                                 atr_multiplier=1.0,
                                 volume_multiplier=1.0)

# ---------------------------
# 4. Merge and Assign Signals with Close Logic
# ---------------------------
# Create a unified list of trades, each marked with its type.
all_trades = []
for trade in long_trades:
    all_trades.append((trade[0], trade[1], 'long'))
for trade in short_trades:
    all_trades.append((trade[0], trade[1], 'short'))

# Sort all trades by the entry index.
all_trades = sorted(all_trades, key=lambda x: x[0])

# Initialize signal columns.
df['long_signal'] = 0
df['short_signal'] = 0
df['close_position'] = 0

# Iterate over the combined trades.
# For each trade, mark the entry in the appropriate column.
# Also mark a close position for the previously open trade.
for i, trade in enumerate(all_trades):
    entry_idx, exit_idx, ttype = trade
    if ttype == 'long':
        df.loc[entry_idx, 'long_signal'] = 1
    else:
        df.loc[entry_idx, 'short_signal'] = 1
    
    # Close the previous trade when a new trade starts.
    # If there is a subsequent trade, mark the close one bar before the next trade’s entry.
    if i < len(all_trades) - 1:
        next_entry = all_trades[i + 1][0]
        # Ensure we mark a close only if it’s after the current entry.
        close_idx = next_entry - 1 if next_entry - 1 > entry_idx else exit_idx
        df.loc[close_idx, 'close_position'] = 1
    else:
        # For the final trade, mark the exit index as the close.
        df.loc[exit_idx, 'close_position'] = 1

# ---------------------------
# 5. Plotting the Signals
# ---------------------------
subset = df[-2000:]

plt.figure(figsize=(12, 6))
plt.plot(subset['datetime'], subset['Close'], label='Close Price', color='blue')

# Plot long entry signals (green upward triangles).
long_signals = subset[subset['long_signal'] == 1]
plt.scatter(long_signals['datetime'], long_signals['Close'],
            marker='^', color='green', s=100, label='Long Entry')

# Plot short entry signals (red downward triangles).
short_signals = subset[subset['short_signal'] == 1]
plt.scatter(short_signals['datetime'], short_signals['Close'],
            marker='v', color='red', s=100, label='Short Entry')

# Plot close positions (black 'x').
close_signals = subset[subset['close_position'] == 1]
plt.scatter(close_signals['datetime'], close_signals['Close'],
            marker='x', color='black', s=100, label='Close Position')

plt.xlabel('Datetime')
plt.ylabel('Close Price')
plt.title('EURUSD Signals: Long, Short, and Close Positions')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# Save Signals
df = df.drop(columns=['ATR', 'vol_ma'])
df.to_csv("EURUSD_1min_sampled_signals.csv", index=False)

# %%
