import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from atrhighsmabreakoutsle.py
def atr_high_sma_breakouts_le(df, atr_period=14, sma_period=100, offset=.5, wide_range_candle=True, volume_increase=True):
    """
    ATRHighSMABreakoutsLE strategy by Ken Calhoun.

    df: pandas.DataFrame: OHLCV data.
    atr_period: int: Period for ATR calculation.
    sma_period: int: Period for SMA calculation.
    offset: float: 'Recorded high' + 'offset' is entry trigger.
    wide_range_candle: bool: True to only trigger if candle height is at least 1.5x the average.
    volume_increase: bool: True to only trigger if the volume has increased since the last candle.
    """

    # Calculate ATR and SMA
    atr = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
    sma = trend.SMAIndicator(df['Close'], window=sma_period).sma_indicator()

    # Create dataframe to store signals
    signals = pd.DataFrame(index=df.index)
    signals['recorded_high'] = df['High'].shift()*(df['High'].shift()==df['High'].rolling(window=atr_period).max())
    signals['current_candle_height'] = df['High'] - df['Low']
    signals['average_candle_height'] = signals['current_candle_height'].rolling(window=atr_period).mean()

    # Conditions
    condition1 = (atr == atr.rolling(window=atr_period).max()) & (df['Close'] > sma)
    condition2 = (df['High'] > (signals['recorded_high'] + offset))
    condition3 = (signals['current_candle_height'] > 1.5 * signals['average_candle_height']) if wide_range_candle else True
    condition4 = (df['Volume'] > df['Volume'].shift()) if volume_increase else True

    # Create signals
    signals['atr_high_sma_breakouts_le_buy_signal'] = 0
    signals.loc[condition1 & condition2 & condition3 & condition4, 'atr_high_sma_breakouts_le_buy_signal'] = 1
    signals.drop(['recorded_high'], axis=1, inplace=True)
    return signals
