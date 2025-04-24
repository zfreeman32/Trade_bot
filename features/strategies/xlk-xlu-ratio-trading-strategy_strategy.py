
import pandas as pd
import numpy as np

# XLK/XLU Ratio Trading Strategy
def xlk_xlu_ratio_signals(xlk_df, xlu_df, window=200):
    signals = pd.DataFrame(index=xlk_df.index)
    
    # Calculate the Ratio
    signals['Ratio'] = xlk_df['Close'] / xlu_df['Close']
    
    # Calculate the 200-day Simple Moving Average of the Ratio
    signals['SMA'] = signals['Ratio'].rolling(window=window).mean()
    
    # Generate Trading Signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['Ratio'] > signals['SMA']), 'signal'] = 'long'
    signals.loc[(signals['Ratio'] < signals['SMA']), 'signal'] = 'short'
    
    return signals[['Ratio', 'SMA', 'signal']]