
import pandas as pd
import numpy as np
from ta import momentum, trend

# RSI and Moving Average Strategy
def rsi_moving_average_signals(stock_df, rsi_period=14, rsi_overbought=70, rsi_oversold=30, sma_period=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    delta = stock_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    
    rs = gain / loss
    signals['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Average
    signals['SMA'] = stock_df['Close'].rolling(window=sma_period).mean()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['RSI'] < rsi_oversold) & (stock_df['Close'] > signals['SMA']), 'signal'] = 'long'
    signals.loc[(signals['RSI'] > rsi_overbought) & (stock_df['Close'] < signals['SMA']), 'signal'] = 'short'
    
    return signals[['RSI', 'SMA', 'signal']]
