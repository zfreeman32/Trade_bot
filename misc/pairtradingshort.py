import pandas as pd
import numpy as np

def calculate_price_ratio(df, symbol1, symbol2):
    return df[symbol1] / df[symbol2]

def calculate_moving_average(series, window):
    return series.rolling(window).mean()

def pair_trading_short_signals(df, symbol1, symbol2, fast_length, slow_length, mode):
    signals = pd.DataFrame(index=df.index)
    price_ratio = calculate_price_ratio(df, symbol1, symbol2)
    
    ma_fast = calculate_moving_average(price_ratio, window=fast_length)
    ma_slow = calculate_moving_average(price_ratio, window=slow_length)
    
    signals['sell_to_open'] = False
    signals['buy_to_close'] = False
    
    if mode == 'trend_following':
        signals.loc[(ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1)), 'sell_to_open'] = True
        signals.loc[(ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1)), 'buy_to_close'] = True
    
    elif mode == 'mean_reversion':
        signals.loc[(price_ratio < ma_fast) & (price_ratio.shift(1) >= ma_fast.shift(1)), 'sell_to_open'] = True
        signals.loc[(price_ratio > ma_fast) & (price_ratio.shift(1) <= ma_fast.shift(1)), 'buy_to_close'] = True
      
    elif mode == 'trend_and_pullback':
        signals.loc[(price_ratio < ma_fast) & (price_ratio > ma_slow) & (price_ratio.shift(1) >= ma_fast.shift(1)), 'sell_to_open'] = True
        signals.loc[(price_ratio > ma_fast) & (price_ratio < ma_slow) & (price_ratio.shift(1) <= ma_fast.shift(1)), 'buy_to_close'] = True

    return signals
