
import pandas as pd
import numpy as np

# US Dollar Index Trading Strategy
def us_dollar_index_signals(stock_df, short_window=10, long_window=50):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Initialize a signal column
    signals['signal'] = 0
    
    # Generate signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    
    # Create a column for trading signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'  # Buy signal
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short' # Sell signal

    return signals[['trading_signal']]
