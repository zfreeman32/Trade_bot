
import pandas as pd
import numpy as np
from ta import momentum

# KST Oscillator Strategy
def kst_oscillator_signals(stock_df, kst_short_window=10, kst_long_window=30, 
                           kst_smoothing=10, kst_signal_period=9):
    """
    Calculate KST Oscillator signals for trading based on the provided DataFrame of stock prices.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    kst_short_window (int): Short period for KST calculation.
    kst_long_window (int): Long period for KST calculation.
    kst_smoothing (int): Smoothing period after KST calculation.
    kst_signal_period (int): Signal line period for KST.

    Returns:
    pd.DataFrame: DataFrame with KST values and trading signals.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Rate of Change for different periods
    roc1 = stock_df['Close'].pct_change(periods=kst_short_window)
    roc2 = stock_df['Close'].pct_change(periods=20)
    roc3 = stock_df['Close'].pct_change(periods=30)
    roc4 = stock_df['Close'].pct_change(periods=40)

    # Calculate KST values using weighted averages
    kst = (roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4).rolling(window=kst_smoothing).sum()
    signals['KST'] = kst

    # Calculate Signal Line
    signals['KST_Signal'] = signals['KST'].rolling(window=kst_signal_period).mean()

    # Generate trading signals
    signals['kst_signal'] = 'neutral'
    signals.loc[(signals['KST'] > signals['KST_Signal']) & 
                 (signals['KST'].shift(1) <= signals['KST_Signal'].shift(1)), 'kst_signal'] = 'long'
    signals.loc[(signals['KST'] < signals['KST_Signal']) & 
                 (signals['KST'].shift(1) >= signals['KST_Signal'].shift(1)), 'kst_signal'] = 'short'

    return signals[['KST', 'KST_Signal', 'kst_signal']]
