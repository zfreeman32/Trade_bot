import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def mfi_signals(stock_df, window=14):
    """
    Computes MFI overbought/oversold signals.

    Returns:
    A DataFrame with 'mfi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute MFI
    mfi = volume.MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['MFI'] = mfi.money_flow_index()

    # Generate signals
    signals['mfi_buy_signal'] = 0
    signals['mfi_sell_signal'] = 0
    signals.loc[signals['MFI'] > 80, 'mfi_buy_signal'] = 1
    signals.loc[signals['MFI'] < 20, 'mfi_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['MFI'], axis=1, inplace=True)

    return signals

#%% 
# Ease of Movement (EOM) Strategy
