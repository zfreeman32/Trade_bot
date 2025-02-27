import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def ema_signals(stock_df, short_window=12, long_window=26):
    """
    Computes EMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'EMA_Direction_Signal' and 'EMA_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate short-term and long-term EMAs
    ema_short = stock_df['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = stock_df['Close'].ewm(span=long_window, adjust=False).mean()

    # Determine market direction
    signals['EMA_bullish_signal'] = 0
    signals['EMA_bearish_signal'] = 0
    signals.loc[ema_short > ema_long, 'EMA_bullish_signal'] = 1
    signals.loc[ema_short < ema_long, 'EMA_bearish_signal'] = 1

    # Generate buy/sell signals based on EMA crossovers
    signals['EMA_buy_signal'] = 0
    signals['EMA_sell_signal'] = 0
    signals.loc[(ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1)), 'EMA_buy_signal'] = 1
    signals.loc[(ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)), 'EMA_sell_signal'] = 1

    return signals

#%% 
# Ichimoku Cloud Strategy
