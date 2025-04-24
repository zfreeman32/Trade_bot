
import pandas as pd
import numpy as np
from ta import momentum, trend

# CADJPY Trading Strategy
def cadjpy_signals(cadjpy_df, short_window=10, long_window=50):
    signals = pd.DataFrame(index=cadjpy_df.index)
    
    # Calculate Moving Averages
    signals['short_mavg'] = cadjpy_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = cadjpy_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1  # Buy Signal
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1  # Sell Signal
    
    # Create a column for position
    signals['positions'] = signals['signal'].diff()
    
    # Assign trading signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short'

    # Drop unnecessary columns
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'positions'], axis=1, inplace=True)
    
    return signals
