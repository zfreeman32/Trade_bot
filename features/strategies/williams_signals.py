import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def williams_signals(stock_df, lbp=14):
    """
    Computes Williams %R overbought and oversold signals.

    Returns:
    A DataFrame with 'williams_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Williams %R
    williams_r = momentum.WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], lbp)
    signals['WilliamsR'] = williams_r.williams_r()

    # Generate overbought/oversold signals
    signals['williams_buy_signal'] = 0
    signals['williams_sell_signal'] = 0
    signals.loc[signals['WilliamsR'] <= -80, 'williams_buy_signal'] = 1
    signals.loc[signals['WilliamsR'] >= -20, 'williams_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['WilliamsR'], axis=1, inplace=True)

    return signals

#%% 
# Rate of Change (ROC) Overbought/Oversold Strategy
