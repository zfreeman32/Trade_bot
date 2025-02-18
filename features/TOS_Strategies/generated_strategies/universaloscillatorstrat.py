import numpy as np
import pandas as pd

def universal_oscillator_strategy(data, cutoff_length=20, mode='reversal'):
    signals = pd.DataFrame(index=data.index)
    close = np.array(data['Close'])
    osc = ehlers_universal_oscillator(close, cutoff_length)
    signals['osc'] = osc
    
    if mode == 'trend_following':
        signals['universal_oscillator_strategy_signals'] = 'neutral'
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_signals'] = 'long'
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_signals'] = 'short'
    elif mode == 'reversal':
        signals['universal_oscillator_strategy_signals'] = 'neutral'
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_signals'] = 'long'
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_signals'] = 'short'
    
    signals.drop(['osc'], axis=1, inplace=True)
    return signals

def ehlers_universal_oscillator(close, cutoff_length):
    alpha = (np.cos(np.sqrt(2) * np.pi / cutoff_length))**2
    beta = 1 - alpha
    a = (1 - beta / 2) * beta
    b = (1 - alpha) * (1 - beta / 2)
    filt = np.zeros_like(close)
    for i in range(2, close.shape[0]):
        filt[i] = a * (close[i] - 2*close[i-1] + close[i-2]) + 2*alpha*filt[i-1] - (alpha**2)*filt[i-2]
    osc = (filt - np.mean(filt)) / np.std(filt)
    return osc
