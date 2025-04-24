
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Breakout Trading Strategy
def breakout_trading_signals(stock_df, breakout_window=20, retest_window=5):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df['High'].rolling(window=breakout_window).max()
    signals['low'] = stock_df['Low'].rolling(window=breakout_window).min()
    
    signals['breakout'] = np.where(stock_df['Close'] > signals['high'].shift(1), 'long', 
                                   np.where(stock_df['Close'] < signals['low'].shift(1), 'short', 'neutral'))
    
    signals['retest_long'] = np.where((signals['breakout'] == 'long') & 
                                       (stock_df['Close'].shift(1) < signals['high'].shift(1)), 'long', 'neutral')
    signals['retest_short'] = np.where((signals['breakout'] == 'short') & 
                                        (stock_df['Close'].shift(1) > signals['low'].shift(1)), 'short', 'neutral')
    
    signals['final_signal'] = np.where(signals['retest_long'] == 'long', 'long', 
                                        np.where(signals['retest_short'] == 'short', 'short', 'neutral'))
    
    signals.drop(['high', 'low', 'breakout', 'retest_long', 'retest_short'], axis=1, inplace=True)
    
    return signals
