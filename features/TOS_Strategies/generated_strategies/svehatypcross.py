import numpy as np
import pandas as pd
import talib

def typical_price(df):
    return (df['High'] + df['Low'] + df['Close']) / 3

def HA(df):
    HA_Close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    HA_Open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    HA_High = df[['High', 'Open', 'Close']].max(axis=1)
    HA_Low = df[['Low', 'Open', 'Close']].min(axis=1)
    return HA_Open, HA_High, HA_Low, HA_Close

def sve_ha_typ_cross_signals(df, typical_length=14, ha_length=14):
    signals = pd.DataFrame(index=df.index)
    
    tp = typical_price(df)
    tp_ema = talib.EMA(tp, timeperiod=typical_length)
    
    ha_open, ha_high, ha_low, ha_close = HA(df)
    ha_avg = (ha_open + ha_high + ha_low + ha_close) / 4
    ha_ema = talib.EMA(ha_avg, timeperiod=ha_length)

    # Initialize signal column with 'neutral'
    signals['sve_ha_typ_cross_signals'] = 'neutral'

    # Assign 'buy' and 'sell' signals
    signals.loc[(tp_ema > ha_ema) & (tp_ema.shift() < ha_ema.shift()), 'sve_ha_typ_cross_signals'] = 'buy'
    signals.loc[(tp_ema < ha_ema) & (tp_ema.shift() > ha_ema.shift()), 'sve_ha_typ_cross_signals'] = 'sell'

    return signals[['sve_ha_typ_cross_signals']]

