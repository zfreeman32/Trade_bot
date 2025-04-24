
import pandas as pd
import numpy as np

# Orange Juice Trading Strategy
def orange_juice_signals(oj_df, short_window=14, long_window=50):
    signals = pd.DataFrame(index=oj_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = oj_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = oj_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    signals['oj_signal'] = 'neutral'
    signals.loc[(signals['signal'] > 0) & (signals['signal'].shift(1) <= 0), 'oj_signal'] = 'long'
    signals.loc[(signals['signal'] < 1) & (signals['signal'].shift(1) >= 1), 'oj_signal'] = 'short'
    
    # Clean up columns
    signals.drop(['short_mavg', 'long_mavg', 'signal'], axis=1, inplace=True)
    
    return signals
