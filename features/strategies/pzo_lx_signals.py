import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# PriceZoneOscillatorLX Strategy
def pzo_lx_signals(df, length=14, ema_length=60, strategy_name='pzo_lx'):
    # Calculate EMA
    ema = trend.EMAIndicator(df['Close'], ema_length).ema_indicator() 
    
    # Calculate ADX
    adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'], length).adx()
    
    # Calculate Bollinger Bands
    bb = volatility.BollingerBands(df['Close'], window=length)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    
    # Calculate PZO
    pzo = (df['Close'] - ((upper + lower) / 2)) / ((upper - lower) / 2) * 100
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=df.index)
    buy_signal_col = 'pzo_lx_buy_signal'
    sell_signal_col = 'pzo_lx_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0
    
    # Set conditions for Long Exit signals
    signals.loc[(adx > 18) & (pzo > 60) & (pzo < pzo.shift(1)), sell_signal_col] = 1  # PZO above +60 and going down in trending
    signals.loc[(adx > 18) & (df['Close'] < ema) & (pzo < 0), sell_signal_col] = 1    # PZO negative and price below EMA in trending
    signals.loc[(adx < 18) & (pzo.shift(1) > 40) & (pzo < 0) & (df['Close'] < ema), sell_signal_col] = 1  # PZO below zero with prior crossing +40 and price below EMA in non-trending
    signals.loc[(adx < 18) & (pzo.shift(1) < 15) & (pzo > -5) & (pzo < 40), sell_signal_col] = 1    # PZO failed to rise above -40, instead fell below -5 in non-trending
    
    return signals
