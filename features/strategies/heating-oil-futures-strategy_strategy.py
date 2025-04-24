
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Heating Oil Futures Trading Strategy
def heating_oil_futures_signals(stock_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the short-term and long-term moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    
    # Determine trading signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
