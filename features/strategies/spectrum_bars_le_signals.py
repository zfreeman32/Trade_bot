import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from spectrumbarsle.py
# SpectrumBarsLE Strategy
def spectrum_bars_le_signals(df, length=10):
    signals = pd.DataFrame(index=df.index)
    signals['close_shift'] = df['Close'].shift(length)
    
    # Define the SpectrumBarsLE conditions:
    # Close price is greater than that from a specified number of bars ago 
    signals['spectrum_bars_buy_signal'] = np.where(df['Close'] > signals['close_shift'], 1, 0)

    # Drop unnecessary columns
    signals.drop(columns='close_shift', inplace=True)
    
    return signals
