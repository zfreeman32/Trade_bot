
import pandas as pd
import numpy as np
from ta import momentum

# Relative Strength Index (RSI) Strategy
def rsi_signals(stock_df, window=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = momentum.RSIIndicator(stock_df['Close'], window)
    signals['RSI'] = rsi.rsi()
    
    signals['rsi_signal'] = 'neutral'
    # Generate long signals
    signals.loc[(signals['RSI'] < oversold), 'rsi_signal'] = 'long'
    # Generate short signals
    signals.loc[(signals['RSI'] > overbought), 'rsi_signal'] = 'short'
    
    # Optional: Mark exits based on RSI returning to neutral range
    signals.loc[(signals['RSI'] >= oversold) & (signals['RSI'] <= overbought), 'rsi_signal'] = 'neutral'
    
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals
