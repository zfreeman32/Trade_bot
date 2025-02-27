import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from fourdaybreakoutle.py
def four_day_breakout_le_signals(stock_df, average_length=20, pattern_length=4, breakout_amount=0.50):
    # Calculate the Simple Moving Average
    sma = trend.SMAIndicator(stock_df['Close'], window=average_length)
    stock_df['SMA'] = sma.sma_indicator()
    
    # Identify all bullish candles
    stock_df['Bullish'] = np.where(stock_df['Close'] > stock_df['Open'], 1, 0)
    
    # Check if the last four candles are all bullish
    stock_df['Bullish_Count'] = stock_df['Bullish'].rolling(window=pattern_length).sum()
    
    # Identify the high of the highest candle in the pattern
    max_pattern_high = stock_df['High'].rolling(window=pattern_length).max().shift()
    
    # Create an empty signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['four_day_breakout_le_buy_signal'] = 0
    
    # Create a buy signal when conditions are met
    signals.loc[(stock_df['Close'] > stock_df['SMA']) &
                (stock_df['Bullish_Count'] == pattern_length) &
                (stock_df['Close'] > (max_pattern_high + breakout_amount)), 'four_day_breakout_le_buy_signal'] = 1
    
    return signals
