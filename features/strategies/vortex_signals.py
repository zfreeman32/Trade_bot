import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Vortex
def vortex_signals(stock_df, window=14):
    """
    Computes Vortex Indicator trend direction and signals.

    Returns:
    A DataFrame with 'vortex_signal' and 'vortex_direction_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Vortex Indicator values
    vortex = trend.VortexIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Positive'] = vortex.vortex_indicator_pos()
    signals['Negative'] = vortex.vortex_indicator_neg()

    # Generate trading signals based on crossovers
    signals['vortex_buy_signal'] = 0
    signals['vortex_sell_signal'] = 0
    signals.loc[
        (signals['Positive'] > signals['Negative']) & 
        (signals['Positive'].shift(1) <= signals['Negative'].shift(1)), 'vortex_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['Positive'] < signals['Negative']) & 
        (signals['Positive'].shift(1) >= signals['Negative'].shift(1)), 'vortex_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['Positive', 'Negative'], axis=1, inplace=True)

    return signals