
import pandas as pd
import numpy as np
from ta import momentum, trend

# NZD CHF Forex Strategy
def nzd_chf_signals(forex_df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate RSI
    delta = forex_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    signals['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals['nzd_chf_signal'] = 'neutral'
    signals.loc[(signals['RSI'] < rsi_oversold), 'nzd_chf_signal'] = 'long'
    signals.loc[(signals['RSI'] > rsi_overbought), 'nzd_chf_signal'] = 'short'
    
    return signals[['nzd_chf_signal']]
