import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def keltner_channel_strategy(stock_df, window=20, window_atr=10, multiplier=2):
    """
    Computes Keltner Channel breakout signals.

    Returns:
    A DataFrame with 'kc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Keltner Channel bands
    keltner_channel = volatility.KeltnerChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window, window_atr, multiplier)
    signals['kc_upper'] = keltner_channel.keltner_channel_hband()
    signals['kc_lower'] = keltner_channel.keltner_channel_lband()

    # Generate breakout signals
    signals['kc_buy_signal'] = 0
    signals['kc_sell_signal'] = 0
    signals.loc[stock_df['Close'] > signals['kc_upper'], 'kc_buy_signal'] = 1
    signals.loc[stock_df['Close'] < signals['kc_lower'], 'kc_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['kc_upper', 'kc_lower'], axis=1, inplace=True)

    return signals

#%% 
# Chaikin Money Flow (CMF) Strategy
