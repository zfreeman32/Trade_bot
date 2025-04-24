
import pandas as pd
import numpy as np
from ta import volatility

# NR7 Strategy (Narrow Range 7)
def nr7_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate the high and low for the last 7 days
    high_7 = stock_df['High'].rolling(window=7).max()
    low_7 = stock_df['Low'].rolling(window=7).min()

    # Calculate the range
    range_7 = high_7 - low_7

    # Identify the NR7 conditions
    signals['NR7'] = range_7
    signals['signal'] = 'neutral'
    
    # Generate 'long' signal if today's range is the narrowest of the last 7 days
    signals.loc[(stock_df['High'] - stock_df['Low'] == signals['NR7'].rolling(window=7).min()), 'signal'] = 'long'
    
    # Generate 'short' signal if today's range is the widest of the last 7 days
    signals.loc[(stock_df['High'] - stock_df['Low'] == signals['NR7'].rolling(window=7).max()), 'signal'] = 'short'
    
    return signals[['signal']]
