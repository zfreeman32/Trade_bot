import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from movavgtwolinesstrat.py
def mov_avg_two_lines_signals(stock_df, fast_length=5, slow_length=20, average_type='EMA', strategy_name='mov_avg_two_lines'):
    signals = pd.DataFrame(index=stock_df.index)
    price = stock_df['Close']

    # Calculate Fast Moving Average
    if average_type == 'SMA':
        fastMA = talib.SMA(price, timeperiod=fast_length)
    elif average_type == 'WMA':
        fastMA = talib.WMA(price, timeperiod=fast_length)
    elif average_type == 'Wilder':
        fastMA = talib.WILDERS(price, timeperiod=fast_length)
    elif average_type == 'Hull':
        fastMA = talib.WMA(price, timeperiod=fast_length)  # Hull is not directly supported by TA-Lib
    else:
        fastMA = talib.EMA(price, timeperiod=fast_length)

    # Calculate Slow Moving Average
    if average_type == 'SMA':
        slowMA = talib.SMA(price, timeperiod=slow_length)
    elif average_type == 'WMA':
        slowMA = talib.WMA(price, timeperiod=slow_length)
    elif average_type == 'Wilder':
        slowMA = talib.WILDERS(price, timeperiod=slow_length)
    elif average_type == 'Hull':
        slowMA = talib.WMA(price, timeperiod=slow_length)  # Hull is not directly supported by TA-Lib
    else:
        slowMA = talib.EMA(price, timeperiod=slow_length)

    buy_signal_col = 'mov_avg_two_lines_buy_signal'
    sell_signal_col = 'mov_avg_two_lines_sell_signal'
    
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    signals.loc[(fastMA > slowMA) & (fastMA.shift(1) <= slowMA.shift(1)), buy_signal_col] = 1
    signals.loc[(fastMA < slowMA) & (fastMA.shift(1) >= slowMA.shift(1)), sell_signal_col] = 1

    return signals
