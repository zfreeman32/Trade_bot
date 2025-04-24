
import pandas as pd
import numpy as np
from ta import trend, volatility

# BOTZ ETF Trading Strategy
def botz_signals(stock_df, dc_length=20, ema_length=50):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Donchian Channels
    high = stock_df['High'].rolling(window=dc_length).max()
    low = stock_df['Low'].rolling(window=dc_length).min()
    signals['donchian_upper'] = high
    signals['donchian_lower'] = low
    
    # Calculate Exponential Moving Average
    signals['EMA'] = trend.ema_indicator(stock_df['Close'], window=ema_length)
    
    # Generate trading signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['donchian_upper']) & 
                 (stock_df['Close'] > signals['EMA']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['donchian_lower']) & 
                 (stock_df['Close'] < signals['EMA']), 'signal'] = 'short'
    
    return signals[['signal']]
