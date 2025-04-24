
import pandas as pd
import numpy as np

# CMO Absolute Indicator Strategy
def cmo_absolute_indicator_signals(stock_df, window=14, overbought=70, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the CMO Absolute Indicator
    delta = stock_df['Close'].diff()
    sum_gain = np.where(delta > 0, delta, 0).rolling(window=window).sum()
    sum_loss = np.where(delta < 0, -delta, 0).rolling(window=window).sum()
    cmo = 100 * (sum_gain - sum_loss) / (sum_gain + sum_loss)

    signals['CMO'] = cmo
    signals['cmo_signal'] = 'neutral'
    
    # Generate signals based on CMO thresholds
    signals.loc[(signals['CMO'] > overbought), 'cmo_signal'] = 'short'
    signals.loc[(signals['CMO'] < oversold), 'cmo_signal'] = 'long'
    
    return signals
