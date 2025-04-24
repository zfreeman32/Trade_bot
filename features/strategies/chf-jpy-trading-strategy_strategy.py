
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# CHFJPY Trading Strategy
def chfjpy_signals(chfjpy_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=chfjpy_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = chfjpy_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = chfjpy_df['Close'].rolling(window=long_window).mean()
    
    # Create signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)

    # Generate trading signals
    signals['chfjpy_signal'] = 'neutral'
    signals.loc[(signals['signal'].diff() == 1), 'chfjpy_signal'] = 'long'  # Buy signal
    signals.loc[(signals['signal'].diff() == -1), 'chfjpy_signal'] = 'short'  # Sell signal
    
    signals.drop(['short_mavg', 'long_mavg', 'signal'], axis=1, inplace=True)
    
    return signals
