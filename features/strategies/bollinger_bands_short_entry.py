import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from bollingerbandsse.py
def bollinger_bands_short_entry(df, length=20, num_devs_up=2):
    # Calculate Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=df["Close"], window=length, window_dev=num_devs_up)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    # Generate signals
    df['bb_short_entry_signal'] = np.where(df['Close'] < df['bb_bbh'], 1, 0)
    signals = pd.DataFrame(df['bb_short_entry_signal'])
    return signals[['bb_short_entry_signal']]
