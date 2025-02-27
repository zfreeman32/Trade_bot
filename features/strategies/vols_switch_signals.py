import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from volswitch.py
# VolSwitch Strategy
def vols_switch_signals(stock_df, length=14, rsi_length=14, avg_length=50, rsi_overbought_level=70, rsi_oversold_level=30, rsi_average_type=MA_Type.SMA):
    # Calculate volatility using standard deviation
    stock_df['volatility'] = stock_df['Close'].rolling(window=length).std()
    
    # Calculate SMA
    stock_df['SMA'] = talib.MA(stock_df['Close'], timeperiod=avg_length, matype=rsi_average_type)

    # Calculate RSI
    stock_df['RSI'] = talib.RSI(stock_df['Close'], timeperiod=rsi_length)

    # Buy signal based on VolSwitch strategy
    stock_df['Buy_Signal'] = ((stock_df['volatility'].shift(1) > stock_df['volatility']) & 
                              (stock_df['Close'] > stock_df['SMA']) & 
                              (stock_df['RSI'] > rsi_oversold_level))

    # Sell signal based on VolSwitch strategy
    stock_df['Sell_Signal'] = ((stock_df['volatility'].shift(1) < stock_df['volatility']) & 
                               (stock_df['Close'] < stock_df['SMA']) & 
                               (stock_df['RSI'] < rsi_overbought_level))
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['vols_switch_buy_signal'] = 0
    signals['vols_switch_sell_signal'] = 0
    
    # Create signals
    signals.loc[(stock_df['Buy_Signal'] == True), 'vols_switch_buy_signal'] = 1
    signals.loc[(stock_df['Sell_Signal'] == True), 'vols_switch_sell_signal'] = 1
 
    return signals
