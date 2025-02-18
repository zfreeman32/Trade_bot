import pandas as pd
import numpy as np
from ta.momentum import ROCIndicator

def roc_signals(ohlcv_df, signal_length=4, roc_length=2, rms_length=3, use_fm_demodulator=False):
    signals = pd.DataFrame(index=ohlcv_df.index)
    
    price = ohlcv_df['Close']
    roc_indicator = ROCIndicator(price, roc_length)
    momentum_roc = roc_indicator.roc()
    
    if use_fm_demodulator:
        fm_demodulator = momentum_roc.rolling(window=rms_length).apply(lambda x: np.sqrt(np.mean(x**2)))
        signal_var = fm_demodulator.rolling(window=signal_length).mean()
    else:
        signal_var = momentum_roc.rolling(window=signal_length).sum() / signal_length
    
    signals['ROC'] = signal_var
    signals['roc_signal'] = 'neutral'
    signals.loc[(signals['ROC'] > 0) & (signals['ROC'].shift(1) <= 0), 'roc_signal'] = 'long'
    signals.loc[(signals['ROC'] < 0) & (signals['ROC'].shift(1) >= 0), 'roc_signal'] = 'short'

    signals.drop(['ROC'], axis=1, inplace=True)
    return signals

