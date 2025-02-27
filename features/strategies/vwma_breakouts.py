import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# VWMA Breakouts Strategy
def vwma_breakouts(stock_df, vwma_length=50, sma_length=70):
    
    df = pd.DataFrame(index=stock_df.index)
    
    # Create the VWAP indicator object
    vwap = volume.VolumeWeightedAveragePrice(
        high = stock_df['High'],
        low = stock_df['Low'],
        close = stock_df['Close'],
        volume = stock_df['Volume'],
        window = vwma_length
    )
    
    # Calculate the VWMA
    df['VWMA'] = vwap.volume_weighted_average_price()

    # Calculate the SMA
    df['SMA'] = df['VWMA'].rolling(window=sma_length).mean()

    # Define signals
    df['vwma_breakouts_buy_signal'] = 0
    df['vwma_breakouts_sell_signal'] = 0
    df.loc[df['VWMA'] > df['SMA'], 'vwma_breakouts_buy_signal'] = 1
    df.loc[df['VWMA'] < df['SMA'], 'vwma_breakouts_sell_signal'] = 1

    return df[['vwma_breakouts_buy_signal', 'vwma_breakouts_sell_signal']]
