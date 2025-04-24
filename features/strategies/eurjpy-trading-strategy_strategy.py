
import pandas as pd
import numpy as np
from ta import momentum, trend

# EURJPY Trading Strategy
def eurjpy_trading_signals(eurjpy_df, short_window=14, long_window=28):
    signals = pd.DataFrame(index=eurjpy_df.index)
    
    # Calculate Moving Averages
    signals['short_moving_average'] = eurjpy_df['Close'].rolling(window=short_window).mean()
    signals['long_moving_average'] = eurjpy_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['short_moving_average'][short_window:] > signals['long_moving_average'][short_window:], 1, 0
    )
    
    signals['position'] = signals['signal'].diff()
    
    # Assign buy and sell signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
