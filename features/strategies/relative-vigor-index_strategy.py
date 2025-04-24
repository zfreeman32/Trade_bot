
import pandas as pd
import numpy as np
from ta.momentum import RelativeVigorIndex

# Relative Vigor Index (RVI) Trading Strategy
def rvi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    rvi = RelativeVigorIndex(stock_df['Close'], window=window)
    signals['RVI'] = rvi.rvi()
    signals['RVI_Signal'] = 'neutral'
    
    # Generating buy and sell signals based on RVI line crossing
    signals.loc[(signals['RVI'] > 0) & (signals['RVI'].shift(1) <= 0), 'RVI_Signal'] = 'long'   # Buy signal
    signals.loc[(signals['RVI'] < 0) & (signals['RVI'].shift(1) >= 0), 'RVI_Signal'] = 'short'  # Sell signal
    
    return signals[['RVI_Signal']]
