
import pandas as pd
import numpy as np

# Trading Strategy Based on Membership Plans
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Using a simple Moving Average crossover strategy as a placeholder for entry/exit rules
    signals['Short_MA'] = stock_df['Close'].rolling(window=5).mean()  # Short-term moving average
    signals['Long_MA'] = stock_df['Close'].rolling(window=20).mean()   # Long-term moving average

    # Generate signals based on moving average crossover
    signals['signal'] = 0  # Default to neutral
    signals['signal'][5:] = np.where(signals['Short_MA'][5:] > signals['Long_MA'][5:], 1, 0)  # Long signal
    signals['signal'][5:] = np.where(signals['Short_MA'][5:] < signals['Long_MA'][5:], -1, signals['signal'][5:])  # Short signal

    # Map numerical signals to textual signals
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'membership_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'membership_signal'] = 'short'
    
    # Clean up the dataframe to return only the signals we are interested in
    signals = signals[['membership_signal']]
    
    return signals
