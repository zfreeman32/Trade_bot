import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from hacoltstrat.py
def hacolt_signals(data, tema_length=14, ema_length=9, candle_size_factor=0.7):
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index)
    
    # Calculate EMA
    ema = talib.EMA(data['Close'], timeperiod=ema_length)
    
    # Calculate TEMA
    tema = talib.T3(data['Close'], timeperiod=tema_length, vfactor=candle_size_factor)
    
    # Calculate HACOLT using EMA and TEMA
    hacolt = (ema / tema) * 100
    
    # Create a column for HACOLT values
    signals['hacolt'] = hacolt
    
    # Create buy and sell signal columns
    signals['hacolt_buy_signal'] = np.where(hacolt == 100, 1, 0)
    signals['hacolt_sell_signal'] = np.where(hacolt == 0, 1, 0)
    
    return signals
