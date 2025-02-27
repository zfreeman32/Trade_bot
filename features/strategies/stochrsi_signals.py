import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def stochrsi_signals(stock_df, window=14, smooth1=3, smooth2=3):
    """
    Computes StochRSI overbought and oversold signals.

    Returns:
    A DataFrame with 'stochrsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate StochRSI
    stoch_rsi = momentum.StochRSIIndicator(stock_df['Close'], window, smooth1, smooth2)
    signals['StochRSI'] = stoch_rsi.stochrsi()

    # Generate overbought/oversold signals
    signals['stochrsi_overbought_signal'] = 0
    signals['stochrsi_oversold_signal'] = 0
    signals.loc[signals['StochRSI'] >= 0.8, 'stochrsi_overbought_signal'] = 1
    signals.loc[signals['StochRSI'] <= 0.2, 'stochrsi_oversold_signal'] = 1

    # Drop temporary column
    signals.drop(['StochRSI'], axis=1, inplace=True)

    return signals

#%% 
# Commodity Channel Index (CCI) Strategy
