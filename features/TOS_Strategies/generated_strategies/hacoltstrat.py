import pandas as pd
import numpy as np
import talib 

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
    
    # Create empty signal column
    signals['hacolt_signal'] = 'neutral'
    
    # Create long entry, short entry, and long exit signals
    signals.loc[hacolt == 100, 'hacolt_signal'] = 'long_entry'
    signals.loc[hacolt == 0, 'hacolt_signal'] = 'short_entry'
    signals.loc[(hacolt != 100) & (signals['hacolt_signal'].shift() == 'long_entry'), 'hacolt_signal'] = 'long_exit'
    
    return signals


