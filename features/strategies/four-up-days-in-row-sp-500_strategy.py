
import pandas as pd
import numpy as np

# Monthly Trading Strategy Signals
def monthly_trading_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Example criteria for generating signals
    # Assuming we have some quantifiable indicators like moving averages for the strategy
    # Here, we'll use the simple moving average (SMA) for illustrative purposes

    # Calculate short (10-day) and long (50-day) term SMA
    short_window = 10
    long_window = 50
    
    signals['SMA_Short'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['SMA_Long'] = stock_df['Close'].rolling(window=long_window).mean()

    # Create signal conditions
    signals['signal'] = 0
    signals.loc[signals['SMA_Short'] > signals['SMA_Long'], 'signal'] = 1  # Long signal
    signals.loc[signals['SMA_Short'] < signals['SMA_Long'], 'signal'] = -1  # Short signal

    # Convert numerical signal to categorical
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'trading_signal'] = 'short'
    
    # Drop the SMA columns to keep only signals
    signals.drop(['SMA_Short', 'SMA_Long', 'signal'], axis=1, inplace=True)

    return signals
