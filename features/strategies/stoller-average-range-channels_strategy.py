
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

# Stoller Average Range Channel (STARC) Strategy
def starc_signals(stock_df, window=20, atr_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Simple Moving Average (SMA)
    signals['SMA'] = stock_df['Close'].rolling(window=window).mean()
    
    # Calculate Average True Range (ATR)
    atr = AverageTrueRange(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=window)
    signals['ATR'] = atr.average_true_range()
    
    # Calculate the upper and lower STARC bands
    signals['Upper Band'] = signals['SMA'] + (signals['ATR'] * atr_multiplier)
    signals['Lower Band'] = signals['SMA'] - (signals['ATR'] * atr_multiplier)
    
    # Generate signals
    signals['starc_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['Upper Band']), 'starc_signal'] = 'short'
    signals.loc[(stock_df['Close'] < signals['Lower Band']), 'starc_signal'] = 'long'
    
    # Clean up DataFrame
    signals.drop(['SMA', 'ATR', 'Upper Band', 'Lower Band'], axis=1, inplace=True)
    
    return signals
