
import pandas as pd
import numpy as np

# Fisher Transform Strategy
def fisher_transform_signals(stock_df, window=10):
    def fisher_transform(data, window):
        min_val = data['Close'].rolling(window=window).min()
        max_val = data['Close'].rolling(window=window).max()
        value = 0.5 * np.log((1 + (data['Close'] - min_val) / (max_val - min_val)) / 
                              (1 - (data['Close'] - min_val) / (max_val - min_val)))
        return value

    signals = pd.DataFrame(index=stock_df.index)
    signals['Fisher'] = fisher_transform(stock_df, window)
    signals['fisher_signal'] = 'neutral'

    # Generate buy and sell signals
    signals.loc[(signals['Fisher'] > 1) & (signals['Fisher'].shift(1) <= 1), 'fisher_signal'] = 'long'
    signals.loc[(signals['Fisher'] < -1) & (signals['Fisher'].shift(1) >= -1), 'fisher_signal'] = 'short'
    
    return signals
