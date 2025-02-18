import pandas as pd
import numpy as np
from ta.trend import SMAIndicator

def four_day_breakout_le_signals(stock_df, average_length=20, pattern_length=4, breakout_amount=0.50):
    # Calculate the Simple Moving Average
    sma = SMAIndicator(stock_df['Close'], window=average_length)
    stock_df['SMA'] = sma.sma_indicator()
    
    # Identify all bullish candles
    stock_df['Bullish'] = np.where(stock_df['Close'] > stock_df['Open'], 1, 0)
    
    # Check if the last four candles are all bullish
    stock_df['Bullish_Count'] = stock_df['Bullish'].rolling(window=pattern_length).sum()
    
    # Identify the high of the highest candle in the pattern
    max_pattern_high = stock_df['High'].rolling(window=pattern_length).max().shift()
    
    # Create an empty signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0
    
    # Create a signal for when the close price is greater than the SMA, 
    # the last four candles are bullish, and the close price is greater 
    # than the high of the highest candle in the pattern by at least the breakout amount
    signals['Signal'] = np.where((stock_df['Close'] > stock_df['SMA']) &
                                 (stock_df['Bullish_Count'] == pattern_length) &
                                 (stock_df['Close'] > (max_pattern_high + breakout_amount)), 1, 0)
    
    # Create a column for simplified, visual-friendly signals
    signals['four_day_breakout_le_signals'] = 'Neutral'
    signals.loc[signals['Signal'] == 1, 'four_day_breakout_le_signals'] = 'Buy'
    
    # Remove all the other columns
    signals.drop(['Signal'], axis=1, inplace=True)

    return signals
