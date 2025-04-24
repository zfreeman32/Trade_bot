
import pandas as pd
import numpy as np
from ta import momentum

# USDCNY Trading Strategy
def usdcny_signals(usdcny_df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """
    Generates trading signals for the USDCNY currency pair based on RSI.

    Parameters:
    usdcny_df (DataFrame): DataFrame containing 'Close' prices of USDCNY
    rsi_period (int): Period for calculating the RSI
    rsi_overbought (int): Overbought threshold for RSI
    rsi_oversold (int): Oversold threshold for RSI

    Returns:
    DataFrame: DataFrame containing trading signals
    """
    signals = pd.DataFrame(index=usdcny_df.index)
    signals['RSI'] = momentum.RSIIndicator(usdcny_df['Close'], window=rsi_period).rsi()
    signals['usdcny_signal'] = 'neutral'

    # Generate buy signals
    signals.loc[(signals['RSI'] < rsi_oversold), 'usdcny_signal'] = 'long'
    
    # Generate sell signals
    signals.loc[(signals['RSI'] > rsi_overbought), 'usdcny_signal'] = 'short'

    return signals[['usdcny_signal']]
