
import pandas as pd
import numpy as np
from ta import trend

# MACD Histogram Trading Strategy
def macd_histogram_signals(stock_df, macd_slow=26, macd_fast=12, macd_signal=9):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate MACD and MACD Histogram
    macd = trend.MACD(stock_df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
    signals['MACD'] = macd.macd()
    signals['MACD_Signal'] = macd.macd_signal()
    signals['MACD_Hist'] = macd.macd_diff()
    
    # Create signals based on MACD Histogram
    signals['macd_signal'] = 'neutral'
    signals.loc[(signals['MACD_Hist'] > 0) & (signals['MACD_Hist'].shift(1) <= 0), 'macd_signal'] = 'long'
    signals.loc[(signals['MACD_Hist'] < 0) & (signals['MACD_Hist'].shift(1) >= 0), 'macd_signal'] = 'short'
    
    # Define exit signals
    signals['exit_signal'] = np.where(signals['macd_signal'].shift(1) == 'long', 
                                       np.where(stock_df['Close'] > stock_df['Close'].shift(1), 'exit_long', 'holding_long'),
                                       np.where(signals['macd_signal'].shift(1) == 'short', 
                                                np.where(stock_df['Close'] < stock_df['Close'].shift(1), 'exit_short', 'holding_short'), 
                                                'neutral'))
    
    return signals[['macd_signal', 'exit_signal']]
