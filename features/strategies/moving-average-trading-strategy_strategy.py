
import pandas as pd
from ta import trend

# Simple Moving Average (SMA) and Exponential Moving Average (EMA) Strategy
def moving_average_signals(stock_df, short_window=50, long_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Simple Moving Averages
    signals['SMA_short'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['SMA_long'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Calculate the Exponential Moving Averages
    signals['EMA_short'] = stock_df['Close'].ewm(span=short_window, adjust=False).mean()
    signals['EMA_long'] = stock_df['Close'].ewm(span=long_window, adjust=False).mean()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['SMA_short'] > signals['SMA_long']) & (signals['SMA_short'].shift(1) <= signals['SMA_long'].shift(1)), 'signal'] = 'long'
    signals.loc[(signals['SMA_short'] < signals['SMA_long']) & (signals['SMA_short'].shift(1) >= signals['SMA_long'].shift(1)), 'signal'] = 'short'
    
    # Drop the moving averages from the DataFrame
    signals.drop(['SMA_short', 'SMA_long', 'EMA_short', 'EMA_long'], axis=1, inplace=True)
    
    return signals
