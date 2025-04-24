
import pandas as pd
import numpy as np

# Average True Range Percentage (ATRP) Strategy
def atrp_signals(stock_df, atr_window=14, atr_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate True Range
    high_low = stock_df['High'] - stock_df['Low']
    high_close_prev = (stock_df['High'] - stock_df['Close'].shift(1)).abs()
    low_close_prev = (stock_df['Low'] - stock_df['Close'].shift(1)).abs()
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate Average True Range (ATR)
    atr = true_range.rolling(window=atr_window).mean()
    
    # Calculate ATR Percentage (ATRP)
    atrp = (atr / stock_df['Close']) * 100
    
    # Generate signals based on ATRP thresholds
    signals['ATRP'] = atrp
    signals['atrp_signal'] = 'neutral'
    signals.loc[signals['ATRP'] < atr_multiplier, 'atrp_signal'] = 'long'
    signals.loc[signals['ATRP'] > atr_multiplier, 'atrp_signal'] = 'short'
    
    signals.drop(['ATRP'], axis=1, inplace=True)
    return signals
