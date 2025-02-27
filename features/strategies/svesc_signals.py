import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from svesc.py
def svesc_signals(stock_df, length=14, exit_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate hlc3
    signals['hlc3'] = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    
    # Calculate Heikin Ashi ohlc4
    ha_open = (stock_df['Open'].shift() + stock_df['Close'].shift()) / 2
    ha_high = stock_df[['High', 'Low', 'Close']].max(axis=1)
    ha_low = stock_df[['High', 'Low', 'Close']].min(axis=1)
    ha_close = (stock_df['Open'] + ha_high + ha_low + stock_df['Close']) / 4
    signals['ha_ohlc4'] = (ha_open + ha_high + ha_low + ha_close) / 4
    
    # Calculate averages
    signals['average_hlc3'] = signals['hlc3'].rolling(window=length).mean()
    signals['average_ha_ohlc4'] = signals['ha_ohlc4'].rolling(window=length).mean()
    signals['average_close'] = stock_df['Close'].rolling(window=exit_length).mean()
    
    # Conditions to simulate orders
    signals['long_entry'] = np.where(signals['average_hlc3'] > signals['average_ha_ohlc4'], 1, 0)
    signals['short_entry'] = np.where(signals['average_hlc3'] < signals['average_ha_ohlc4'], -1, 0)
    signals['long_exit'] = np.where((signals['short_entry'].shift() == -1) &(stock_df['Close'] > signals['average_close']) & (stock_df['Close'] > stock_df['Open']), 1, 0)
    signals['short_exit'] = np.where((signals['long_entry'].shift() == 1) & (stock_df['Close'] < signals['average_close']) & (stock_df['Close'] < stock_df['Open']), -1, 0)
    
    # Consolidate signals
    signals['orders'] = signals['long_entry'] + signals['short_entry'] + signals['long_exit'] + signals['short_exit']
    
    # Map signal to action
    signals['svesc_buy_signal'] = 0
    signals['svesc_sell_signal'] = 0
    signals.loc[signals['orders'] > 0 , 'svesc_buy_signal'] = 1
    signals.loc[signals['orders'] < 0 , 'svesc_sell_signal'] = 1
    
    # Drop unnecessary columns
    signals.drop(columns=['hlc3', 'ha_ohlc4', 'average_hlc3', 'average_ha_ohlc4', 'average_close', 'long_entry', 'short_entry', 'long_exit', 'short_exit', 'orders'], inplace=True)
    
    return signals
