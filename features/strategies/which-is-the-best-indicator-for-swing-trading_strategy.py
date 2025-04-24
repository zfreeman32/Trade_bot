
import pandas as pd
import numpy as np
from ta.momentum import WilliamsRIndicator

# Williams %R Strategy
def williams_r_signals(stock_df, window=14, overbought_level=-20, oversold_level=-80):
    signals = pd.DataFrame(index=stock_df.index)
    williams_r = WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)
    signals['Williams %R'] = williams_r.williams_r()
    signals['williams_r_signal'] = 'neutral'
    
    # Generate long signals
    signals.loc[(signals['Williams %R'] < oversold_level), 'williams_r_signal'] = 'long'
    
    # Generate short signals
    signals.loc[(signals['Williams %R'] > overbought_level), 'williams_r_signal'] = 'short'
    
    return signals[['williams_r_signal']]
