
import pandas as pd
import numpy as np

# Hurst Exponent Trading Strategy
def hurst_exponent_signals(stock_df, window=20):
    def hurst_exponent(ts):
        lags = range(2, 100)
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        log_lags = np.log(lags)
        log_tau = np.log(tau)
        return np.polyfit(log_lags, log_tau, 1)[0]

    signals = pd.DataFrame(index=stock_df.index)
    signals['Hurst'] = stock_df['Close'].rolling(window).apply(hurst_exponent)
    signals['hurst_signal'] = 'neutral'
    signals.loc[signals['Hurst'] > 0.5, 'hurst_signal'] = 'long'
    signals.loc[signals['Hurst'] < 0.5, 'hurst_signal'] = 'short'
    return signals
