
import pandas as pd
import numpy as np

# Adaptive Laguerre Filter Strategy
def adaptive_laguerre_filter_signals(stock_df, gamma=0.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Adaptive Laguerre Filter
    laguerre = np.zeros(len(stock_df))
    for i in range(4, len(stock_df)):
        laguerre[i] = (1 - gamma) * (laguerre[i-1] + gamma * stock_df['Close'].iloc[i-1]) + gamma * laguerre[i-1]

    signals['Laguerre'] = laguerre
    signals['laguerre_signal'] = 'neutral'
    
    # Generate signals
    signals.loc[(signals['Laguerre'] > stock_df['Close']), 'laguerre_signal'] = 'short'
    signals.loc[(signals['Laguerre'] < stock_df['Close']), 'laguerre_signal'] = 'long'
    
    return signals[['laguerre_signal']]
