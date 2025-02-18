# Import Necessary Libraries
import pandas as pd
import talib

# Moving Average Two Lines strategy
def mov_avg_two_lines_signals(stock_df, fast_length=5, slow_length=20, average_type='EMA'):
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
        fastMA = talib.WMA(price, timeperiod=fast_length) # Hull is not directly supported by TA-Lib
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
        slowMA = talib.WMA(price, timeperiod=slow_length) # Hull is not directly supported by TA-Lib
    else:
        slowMA = talib.EMA(price, timeperiod=slow_length)

    signals['FastMA'] = fastMA
    signals['SlowMA'] = slowMA
    signals['mov_avg_two_lines_signals'] = 'neutral'
    signals.loc[(signals['FastMA'] > signals['SlowMA']) & (signals['FastMA'].shift(1) <= signals['SlowMA'].shift(1)), 'mov_avg_two_lines_signals'] = 'long'
    signals.loc[(signals['FastMA'] < signals['SlowMA']) & (signals['FastMA'].shift(1) >= signals['SlowMA'].shift(1)), 'mov_avg_two_lines_signals'] = 'short'

    signals.drop(['FastMA', 'SlowMA'], axis=1, inplace=True)
    return signals
