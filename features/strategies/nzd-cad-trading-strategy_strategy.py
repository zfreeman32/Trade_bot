
import pandas as pd
import numpy as np
from ta import momentum, trend

# NZD/CAD Trading Strategy
def nzd_cad_signals(forex_df, short_window=14, long_window=50):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate the short-term and long-term moving averages
    signals['short_mavg'] = forex_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = forex_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals based on moving average crossover
    signals['signal'] = 0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1  # Long position
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1  # Short position
    
    # Create a column for positions
    signals['position'] = signals['signal'].diff()
    
    # Generate trading signals
    signals['nzd_cad_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'nzd_cad_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'nzd_cad_signal'] = 'short'
    
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'position'], axis=1, inplace=True)
    
    return signals
