import pandas as pd
import talib
from talib import MA_Type

def volty_expan_close_lx(df, num_atrs=2, length=14, ma_type='SMA'):
    df = df.reset_index()  # ensure that the Dataframe index is sequential
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=length)

    # Get the MA_Type based on the average type input
    if ma_type == 'SMA':
        ma_type = MA_Type.SMA
    elif ma_type == 'EMA':
        ma_type = MA_Type.EMA
    elif ma_type == 'WMA':
        ma_type = MA_Type.WMA
    elif ma_type == 'DEMA':
        ma_type = MA_Type.DEMA
    elif ma_type == 'TEMA':
        ma_type = MA_Type.TEMA
    elif ma_type == 'TRIMA':
        ma_type = MA_Type.TRIMA
    elif ma_type == 'KAMA':
        ma_type = MA_Type.KAMA
    elif ma_type == 'MAMA':
        ma_type = MA_Type.MAMA
    else:
        ma_type = MA_Type.SMA

    atr_ma = talib.MA(atr, timeperiod=length, matype=ma_type)

    df['volty_expan_close_lx_signal'] = 0
    for i in range(length, len(df)):
        if df.loc[i, 'Low'] < (df.loc[i-1, 'Close'] - num_atrs * atr_ma[i]):
            df.loc[i, 'volty_expan_close_lx_signal'] = -1 if df.loc[i, 'Open'] < (df.loc[i-1, 'Close'] - num_atrs * atr_ma[i]) else 0

    return df[['volty_expan_close_lx_signal']]

