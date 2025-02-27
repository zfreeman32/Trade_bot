import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from camarillapointsstrat.py
def camarilla_strategy(stock_df):
    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']

    # Calculate the pivot point.
    pivot_point = (high + low + close) / 3

    # Calculate the support and resistance levels based on the pivot point.
    ranges = high - low
    r1 = close + ranges * 1.1 / 12
    s1 = close - ranges * 1.1 / 12
    r2 = close + ranges * 1.1 / 6
    s2 = close - ranges * 1.1 / 6
    r3 = close + ranges * 1.1 / 4
    s3 = close - ranges * 1.1 / 4
    r4 = close + ranges * 1.1 / 2
    s4 = close - ranges * 1.1 / 2

    stock_df['PP'] = pivot_point
    stock_df['R1'] = r1
    stock_df['R2'] = r2
    stock_df['R3'] = r3
    stock_df['R4'] = r4
    stock_df['S1'] = s1
    stock_df['S2'] = s2
    stock_df['S3'] = s3
    stock_df['S4'] = s4

    signals = pd.DataFrame(index=stock_df.index)
    signals['camarilla_buy_signal'] = 0
    signals['camarilla_sell_signal'] = 0
  
    # Long entry
    signals.loc[((stock_df.Open < stock_df.S3) & (stock_df.Close > stock_df.S3)), 'camarilla_buy_signal'] = 1
    # Short entry
    signals.loc[((stock_df.Open > stock_df.R3) & (stock_df.Close < stock_df.R3)), 'camarilla_sell_signal'] = 1

    return signals[['camarilla_buy_signal', 'camarilla_sell_signal']]
