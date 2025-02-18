
import pandas as pd
import numpy as np
import talib

def LongHaul_strategy(stock_df, fast_length=5, slow_length=25, rsi_length=14, 
                      rsi_over_sold_level=30, rsi_over_bought_level=70,
                      rsi_average_type='SMA', high_length=5):   

    # Calculate MA
    if rsi_average_type == 'SMA':
        stock_df['FastMA'] = talib.SMA(stock_df['Close'].values, fast_length)
        stock_df['SlowMA'] = talib.SMA(stock_df['Close'].values, slow_length)

    elif rsi_average_type == 'EMA':
        stock_df['FastMA'] = talib.EMA(stock_df['Close'].values, fast_length)
        stock_df['SlowMA'] = talib.EMA(stock_df['Close'].values, slow_length)

    # Calculate RSI
    stock_df['RSI'] = talib.RSI(stock_df['Close'].values, timeperiod=rsi_length)

    # Get the highest high over high_length period
    stock_df['High_Length_High'] = stock_df['High'].rolling(window=high_length).max()

    # Long location
    stock_df['Long'] = ((stock_df['RSI'] < rsi_over_sold_level) & 
                        (stock_df['Close'].shift() < stock_df['SlowMA'].shift()) &
                        (stock_df['Close'] > stock_df['SlowMA']) &
                        (stock_df['Close'] > stock_df['High_Length_High'].shift()))
    
    # Exit location
    stock_df['Exit'] = ((stock_df['Close'] < stock_df['FastMA']) | 
                        (stock_df['Close'].shift(3) < stock_df['Close'].shift(4)) &
                        (stock_df['Close'].shift(2) < stock_df['Close'].shift(3)) &
                        (stock_df['Close'].shift(1) < stock_df['Close'].shift(2)) &
                        (stock_df['Close'] < stock_df['Close'].shift(1)))


    stock_df['LongHaul_strategy_signals'] = np.where(stock_df['Long'], 1, np.nan)
    stock_df['LongHaul_strategy_signals'] = np.where(stock_df['Exit'], 0, stock_df['LongHaul_strategy_signals'])
    stock_df['LongHaul_strategy_signals'] = stock_df['LongHaul_strategy_signals'].ffill().fillna(0)

    return stock_df['LongHaul_strategy_signals']
