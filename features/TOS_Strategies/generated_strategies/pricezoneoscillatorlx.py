import pandas as pd
import numpy as np
from ta import trend, volatility

# PriceZoneOscillatorLX Strategy
def pzo_signals(df, length=14, ema_length=60):
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
    signals['pzo_lx_signals'] = pzo
    signals['pzo_lx_trend'] = 'neutral'
    
    # Set conditions for Long Exit signals
    conditions = [
        (adx > 18) & (pzo > 60) & (pzo < pzo.shift(1)),  # PZO above +60 and going down in trending 
        (adx > 18) & (df['Close'] < ema) & (pzo < 0),    # PZO negative and price below EMA in trending 
        (adx < 18) & (pzo.shift(1) > 40) & (pzo < 0) & (df['Close'] < ema),     # PZO below zero with prior crossing +40 and price below EMA in non-trending
        (adx < 18) & (pzo.shift(1) < 15) & (pzo > -5) & (pzo < 40)    # PZO failed to rise above -40, instead fell below -5 in non-trending
    ]
    
    choices = ['long_exit'] * 4
    
    # Apply conditions and choices
    signals['pzo_lx_trend'] = np.select(conditions, choices, 'neutral')
    
    return signals
