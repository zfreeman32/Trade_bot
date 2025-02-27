import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from trailingstopsx.py
def trailing_stop_sx_signals(stock_df, trail_stop=1.0, offset_type='percent'):
    signals = pd.DataFrame(index=stock_df.index)
    entry_price = stock_df['Close'].shift()  # Use the closing price as the entry price

    if offset_type == 'value':
        stop_price = entry_price + trail_stop
    elif offset_type == 'percent':
        stop_price = entry_price * (1 + trail_stop / 100)
    elif offset_type == 'tick':
        tick_size = 0.01  # for simplicity, assuming tick size is 0.01
        stop_price = entry_price + trail_stop * tick_size
    else:
        raise ValueError(f'Invalid offset_type "{offset_type}". Choose from "value", "percent", "tick"')

    signals['TrailingStop'] = np.maximum.accumulate(stop_price) # Accumulates the maximum stop price
    signals['trailing_stop_sx_buy_signal'] = 0
    signals.loc[(stock_df['High'] > signals['TrailingStop']), 'trailing_stop_sx_buy_signal'] = 1
    signals.drop(['TrailingStop'], axis=1, inplace=True)

    return signals[['trailing_stop_sx_buy_signal']]
