import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from bollingerbandsle.py
def bollinger_bands_le_signals(data, length=20, num_devs_dn=2.0):
    # Instantiate Bollinger Bands indicator
    indicator_bb = volatility.BollingerBands(close=data['Close'], window=length, window_dev=num_devs_dn)
    
    # Create DataFrame for signals
    signals = pd.DataFrame(index=data.index)
    signals['lower'] = indicator_bb.bollinger_lband()  # Calculate lower band
    signals['close'] = data['Close']  # Track closing prices
    signals['bollinger_bands_le_buy_signal'] = 0 # Default all signals to 0.0

    # Generate 'Long Entry' signal where price crosses above lower band
    signals.loc['bollinger_bands_le_buy_signal'][signals['close'] > signals['lower'].shift(1)] = 1
    signals.drop(columns=['close', 'lower'], inplace=True)
    # Return only the signal column
    return signals
