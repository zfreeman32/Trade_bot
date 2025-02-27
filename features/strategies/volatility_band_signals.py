import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from volatilityband.py
def volatility_band_signals(df, price_col='Close', average_length=20, deviation=2, low_band_adjust=0.5):
    
    # Extract the actual price column
    price = df[price_col]

    # Calculate SMA of price data
    mid_line = price.rolling(average_length).mean()
    
    # Create Bollinger Bands Indicator
    indicator_bb = volatility.BollingerBands(close=price, window=average_length, window_dev=deviation)

    # Get upper and lower bands
    df['upper_band'] = indicator_bb.bollinger_hband()
    df['lower_band'] = mid_line - ((indicator_bb.bollinger_hband() - mid_line) * low_band_adjust)

    # Create signals DataFrame
    signals = pd.DataFrame(index=df.index)
    signals['volatility_band_buy_signal'] = 0
    signals['volatility_band_sell_signal'] = 0

    # Create buy and sell signals
    signals.loc[price > df['upper_band'], 'volatility_band_sell_signal'] = 1
    signals.loc[price < df['lower_band'], 'volatility_band_buy_signal'] = 1

    return signals
