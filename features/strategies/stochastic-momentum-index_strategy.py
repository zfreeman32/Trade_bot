
import pandas as pd
import numpy as np

# Stochastic Momentum Index (SMI) Strategy
def smi_signals(stock_df, window=14, smooth_window=3):
    """
    Generates trading signals based on the Stochastic Momentum Index (SMI).

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with 'Close', 'High', and 'Low' columns.
    window (int): The window length for the SMI calculation.
    smooth_window (int): The smoothing window for the SMI line.

    Returns:
    pd.DataFrame: DataFrame with SMI values and associated trading signals.
    """
    # Calculate the SMI
    high_max = stock_df['High'].rolling(window=window).max()
    low_min = stock_df['Low'].rolling(window=window).min()
    SMI = ((stock_df['Close'] - ((high_max + low_min) / 2)) / ((high_max - low_min) / 2)) * 100

    # Smooth the SMI
    smi = SMI.rolling(window=smooth_window).mean()

    signals = pd.DataFrame(index=stock_df.index)
    signals['SMI'] = smi
    signals['smi_signal'] = 'neutral'

    # Trading signals based on SMI thresholds
    signals.loc[(signals['SMI'] > 40) & (signals['SMI'].shift(1) <= 40), 'smi_signal'] = 'long'
    signals.loc[(signals['SMI'] < -40) & (signals['SMI'].shift(1) >= -40), 'smi_signal'] = 'short'

    return signals
