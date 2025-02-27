import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from voltyexpancloselx.py
import talib
from talib import MA_Type

def volty_expan_close_lx(df, num_atrs=2, length=14, ma_type='SMA'):
    df = df.copy()  # Avoid modifying the original DataFrame

    # Compute ATR
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=length)

    # Map string input to TALib's MA_Type
    ma_type_mapping = {
        'SMA': MA_Type.SMA, 'EMA': MA_Type.EMA, 'WMA': MA_Type.WMA,
        'DEMA': MA_Type.DEMA, 'TEMA': MA_Type.TEMA, 'TRIMA': MA_Type.TRIMA,
        'KAMA': MA_Type.KAMA, 'MAMA': MA_Type.MAMA
    }
    ma_type = ma_type_mapping.get(ma_type, MA_Type.SMA)

    # Compute ATR moving average
    df['ATR_MA'] = talib.MA(df['ATR'], timeperiod=length, matype=ma_type)

    # Initialize signal column
    df['volty_expan_close_lx_sell_signal'] = 0

    # Avoid looping; use vectorized calculation
    condition = (df['Low'] < (df['Close'].shift(1) - num_atrs * df['ATR_MA'])) & \
                (df['Open'] < (df['Close'].shift(1) - num_atrs * df['ATR_MA']))
    
    df.loc[condition, 'volty_expan_close_lx_sell_signal'] = 1

    # Drop intermediate columns if not needed
    df.drop(columns=['ATR', 'ATR_MA'], inplace=True)

    return df[['volty_expan_close_lx_sell_signal']]
