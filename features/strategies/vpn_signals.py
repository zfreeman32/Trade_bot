import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from vpnstrat.py
def vpn_signals(df, length=14, ema_length=14, average_length=10, rsi_length=14, volume_average_length=50, highest_length=14, atr_length=14, factor=1, critical_value=10, rsi_max_overbought_level=90, num_atrs=1, average_type='simple'):
    vwap = volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=average_length)

    # Calculate VPN 
    vpn = (df['Volume'] * factor * np.where(df['Close'].diff() > 0, 1, -1)).ewm(span=length).mean() / df['Volume'].ewm(span=length).mean()
    vpn_average = vpn.ewm(span=average_length).mean() if average_type == 'exponential' else vpn.rolling(window=average_length).mean()
    
    # Create signal column
    signals = pd.DataFrame(index=df.index)
    signals['vpn_buy_signal'] = 0
    signals['vpn_sell_signal'] = 0

    # Buy condition
    buy_condition = (
        (vpn > critical_value) & 
        (df['Close'] > vwap.vwap) &  # Fixed incorrect function call
        (df['Volume'].rolling(window=volume_average_length).mean().diff() > 0) & 
        (momentum.RSIIndicator(close=df['Close'], window=rsi_length).rsi() < rsi_max_overbought_level)
    )

    # Sell condition
    sell_condition = (
        (vpn < vpn_average) & 
        (df['Close'] < df['Close'].rolling(window=highest_length).max() - num_atrs * 
         volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=atr_length).average_true_range())
    )

    # Assign buy and sell signals
    signals.loc[buy_condition, 'vpn_buy_signal'] = 1
    signals.loc[sell_condition, 'vpn_sell_signal'] = 1

    return signals
