import pandas as pd
import numpy as np
from ta.volatility import BollingerBands

def volatility_band_signals(df, price_col='Close', average_length=20, deviation=2, low_band_adjust=0.5):
    
    # Extract the actual price column
    price = df[price_col]

    # Calculate SMA of price data
    mid_line = price.rolling(average_length).mean()
    
    # Create Bollinger Bands Indicator
    indicator_bb = BollingerBands(close=price, window=average_length, window_dev=deviation)

    # Get upper and lower bands
    df['upper_band'] = indicator_bb.bollinger_hband()
    df['lower_band'] = mid_line - ((indicator_bb.bollinger_hband() - mid_line) * low_band_adjust)

    # Create signals DataFrame
    signals = pd.DataFrame(index=df.index)
    signals['volatility_band_signals'] = 'neutral'

    # Create buy and sell signals
    signals.loc[price > df['upper_band'], 'volatility_band_signals'] = 'sell'
    signals.loc[price < df['lower_band'], 'volatility_band_signals'] = 'buy'

    return signals


