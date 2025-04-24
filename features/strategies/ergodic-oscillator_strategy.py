
import pandas as pd
import numpy as np
from ta import momentum

# Ergodic Oscillator Strategy
def ergodic_oscillator_signals(stock_df, window_slow=13, window_fast=25, signal_window=9):
    """
    Generate trading signals based on the Ergodic Oscillator strategy.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with a 'Close' column.
    window_slow (int): The slow window for the True Strength Index calculation.
    window_fast (int): The fast window for the True Strength Index calculation.
    signal_window (int): The window for the signal line.

    Returns:
    pd.DataFrame: A DataFrame with trading signals.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the True Strength Index (TSI)
    price = stock_df['Close']
    delta = price.diff()
    ema1 = delta.ewm(span=window_fast, adjust=False).mean()
    ema2 = ema1.ewm(span=window_slow, adjust=False).mean()
    
    tsi = (ema1 / ema2) * 100
    
    # Calculate the Ergodic Oscillator
    signal_line = tsi.ewm(span=signal_window, adjust=False).mean()
    ergodic_oscillator = tsi - signal_line
    
    # Generate signals
    signals['Ergodic_Oscillator'] = ergodic_oscillator
    signals['Signal'] = 'neutral'
    signals.loc[(signals['Ergodic_Oscillator'] > 0) & (signals['Ergodic_Oscillator'].shift(1) <= 0), 'Signal'] = 'long'
    signals.loc[(signals['Ergodic_Oscillator'] < 0) & (signals['Ergodic_Oscillator'].shift(1) >= 0), 'Signal'] = 'short'
    
    return signals[['Signal']]
