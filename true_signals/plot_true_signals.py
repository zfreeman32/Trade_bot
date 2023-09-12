#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("../data/SPY.csv")
data.drop(columns=['Date', 'Adj Close'], inplace=True)
signal = pd.read_csv("../data/SPY_true_signals.csv")
signal.drop(columns=['Close'], inplace=True)
# Encode the 'signals' column
signal['signals'] = signal['signals'].map({'long': 1, 'short': -1, 0: 0})
df = pd.concat([data, signal], axis=1)
df = df.fillna(0)
df = df.replace('nan', 0)

closepos_column = df.columns[-1]
closepos_values = df[closepos_column]

signal_column = df.columns[-2]
signal_values = df[signal_column]

# Count the number of 1s and -1s in signal_values
signal_counts = signal_values.value_counts()

# Print the results
print("Number of 1s and -1s in signal_values:")
print(signal_counts)

# %%
# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the 'Close' price data
ax.plot(df['Close'], label='Close Price', color='blue')

# Plot green arrows for buy signals (signal_values == 1)
buy_signals = df[df['signals'] == 1]
ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal')

# Plot red arrows for sell signals (signal_values == -1)
sell_signals = df[df['signals'] == -1]
ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal')

# Set labels and legend
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title('SPY Stock Close Price with Buy/Sell Signals')
ax.legend()

# Show the plot
plt.show()

# %%
