import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from insidebarle.py
def inside_bar_le(df):
    # Create a copy of the DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create a new column for the Long Entry signals
    signals['inside_bar_le_buy_signal'] = 0

    # Calculate the Inside Bar condition
    signals['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    
    # Generate the Long Entry signal condition
    signals.loc[signals['Inside_Bar'] & (df['Close'] > df['Open']), 'inside_bar_le_buy_signal'] = 1
    signals.drop(['Inside_Bar'], axis=1, inplace=True)
    return signals['inside_bar_le_buy_signal']
