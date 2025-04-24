
import pandas as pd
import numpy as np

# Intraday Trading Strategy
def intraday_trading_signals(stock_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculating short and long moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Generating signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0
    )
    
    signals['positions'] = signals['signal'].diff()
    
    # Defining trade signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short'
    
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'positions'], axis=1, inplace=True)
    return signals
