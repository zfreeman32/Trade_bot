import pandas as pd
import numpy as np
from ta import trend

def calculate_moving_avg(close, length=10, average_type='simple'):
    if average_type == 'simple':
        return close.rolling(window=length).mean()
    elif average_type == 'exponential':
        return close.ewm(span=length, adjust=False).mean()
    elif average_type == 'wilder':
        return trend.EMAIndicator(close=close, window=length, fillna=False).ema_indicator()
    elif average_type == 'hull':
        return trend.WMAIndicator(close=close, window=length, fillna=False).wma_indicator()
    else: 
        return None

def trend_following_signals(stock_df, length=10, average_type='simple', entry_percent=3.0, exit_percent=4.0):
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close']
    moving_avg = calculate_moving_avg(close, length, average_type)
    
    # Define conditions
    long_entry = close > moving_avg * (1 + entry_percent / 100)
    long_exit = close < moving_avg * (1 - exit_percent / 100)
    short_entry = close < moving_avg * (1 - entry_percent / 100)
    short_exit = close > moving_avg * (1 + exit_percent / 100)

    # Create two separate signal columns with function name included
    signals['trend_following_buy_signals'] = 'neutral'
    signals['trend_following_sell_signals'] = 'neutral'

    # Assign 'buy' signals
    signals.loc[long_entry | short_exit, 'trend_following_buy_signals'] = 'buy'

    # Assign 'sell' signals
    signals.loc[long_exit | short_entry, 'trend_following_sell_signals'] = 'sell'

    return signals

