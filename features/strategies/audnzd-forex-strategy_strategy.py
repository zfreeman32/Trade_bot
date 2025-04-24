
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# AUDNZD Trading Strategy
def audnzd_signals(audnzd_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=audnzd_df.index)
    
    # Calculate moving averages
    signals['Short_MA'] = audnzd_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = audnzd_df['Close'].rolling(window=long_window).mean()
    
    # Determine trading signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    
    # Define the signal types
    signals['audnzd_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'audnzd_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'audnzd_signal'] = 'short'
    
    # Drop unnecessary columns
    signals.drop(['Short_MA', 'Long_MA', 'signal', 'positions'], axis=1, inplace=True)
    
    return signals
