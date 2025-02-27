import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from vhftrend.py
# VHF Trend Strategy
def vhf_signals(stock_df, length=14, lag=14, avg_length=14, trend_level=0.5, max_level=0.75, crit_level=0.25, mult=2, avg_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the VHF indicator
    vhf = (np.max(stock_df['High'] - stock_df['Low']) / (stock_df['Close'].diff().abs().rolling(length).sum())) * 100
    
    # Get Lowest VHF value in previous period
    min_vhf = vhf.rolling(lag).min()

    # Get Moving Average
    if avg_type.lower() == 'simple':
        ma = stock_df['Close'].rolling(avg_length).mean()
    elif avg_type.lower() == 'exponential':
        ma = stock_df['Close'].ewm(span=avg_length).mean()
    elif avg_type.lower() == 'weighted':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    elif avg_type.lower() == 'wilders':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    elif avg_type.lower() == 'hull':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    else:
        raise ValueError("Invalid average type")

    signals['vhf_buy_signal'] = 0
    signals['vhf_sell_signal'] = 0

    # Generate Buy and Sell signals
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] > ma), 'vhf_buy_signal'] = 1
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] > ma), 'vhf_buy_signal'] = 1
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] < ma), 'vhf_sell_signal'] = 1
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] < ma), 'vhf_sell_signal'] = 1

    return signals
