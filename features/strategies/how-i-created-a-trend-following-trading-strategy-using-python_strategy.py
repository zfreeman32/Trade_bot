
import pandas as pd
import numpy as np
from ta import momentum

# Example strategy: Williams %R Strategy
def williams_r_signals(stock_df, period=14, upper_threshold=-20, lower_threshold=-80):
    signals = pd.DataFrame(index=stock_df.index)
    williams_r = momentum.WilliamsRIndicator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=period)
    signals['Williams %R'] = williams_r.williams_r()
    signals['williams_signal'] = 'neutral'
    
    signals.loc[(signals['Williams %R'] < upper_threshold) & (signals['Williams %R'].shift(1) >= upper_threshold), 'williams_signal'] = 'long'
    signals.loc[(signals['Williams %R'] > lower_threshold) & (signals['Williams %R'].shift(1) <= lower_threshold), 'williams_signal'] = 'short'
    
    signals.drop(['Williams %R'], axis=1, inplace=True)
    return signals
