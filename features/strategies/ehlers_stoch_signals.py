import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Ehlers Decycler Oscillator Signal Strategy
def ehlers_stoch_signals(stock_df, length1=10, length2=20):
    signals = pd.DataFrame(index=stock_df.index)
    price = stock_df['Close']
    
    # High-pass filter function
    def hp_filter(src, length):
        pi = 2 * np.arcsin(1)
        twoPiPrd = 0.707 * 2 * pi / length
        a = (np.cos(twoPiPrd) + np.sin(twoPiPrd) - 1) / np.cos(twoPiPrd)
        hp = np.zeros_like(src)
        for i in range(2, len(src)):
            hp[i] = ((1 - a / 2) ** 2) * (src[i] - 2 * src[i-1] + src[i-2]) + \
                2 * (1 - a) * hp[i-1] - ((1 - a) ** 2) * hp[i-2]
        return hp
    
    # Compute the Ehlers Decycler Oscillator
    hp1 = hp_filter(price, length1)
    hp2 = hp_filter(price, length2)
    dec = hp2 - hp1
    
    # Compute signal strength
    slo = dec - np.roll(dec, 1)
    sig = np.where(slo > 0, np.where(slo > np.roll(slo, 1), 2, 1),
                   np.where(slo < 0, np.where(slo < np.roll(slo, 1), -2, -1), 0))
    
    # Generate separate buy and sell signals
    signals['ehlers_stoch_buy_signal'] = np.where((sig == 1) | (sig == 2), 1, 0)
    signals['ehlers_stoch_sell_signal'] = np.where((sig == -1) | (sig == -2), 1, 0)
    
    return signals
