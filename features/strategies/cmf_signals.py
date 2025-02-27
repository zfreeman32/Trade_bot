import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def cmf_signals(stock_df, window=20, threshold=0.1):
    """
    Computes CMF trend signals.

    Returns:
    A DataFrame with 'cmf_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute CMF
    cmf = volume.ChaikinMoneyFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['CMF'] = cmf.chaikin_money_flow()

    # Generate signals
    signals['cmf_buy_signal'] = 0
    signals['cmf_sell_signal'] = 0
    signals.loc[signals['CMF'] > threshold, 'cmf_buy_signal'] = 1
    signals.loc[signals['CMF'] < -threshold, 'cmf_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['CMF'], axis=1, inplace=True)

    return signals

#%% 
# Money Flow Index (MFI) Strategy
