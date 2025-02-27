import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

def cci_signals(stock_df, window=20, constant=0.015, overbought=100, oversold=-100):
    """
    Computes CCI trend direction and trading signals.

    Returns:
    A DataFrame with 'cci_direction' and 'cci_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create CCI Indicator
    cci = trend.CCIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window, constant)
    signals['CCI'] = cci.cci()

    # Determine market direction
    signals['cci_bullish_signal'] = 0
    signals['cci_bearish_signal'] = 0
    signals.loc[signals['CCI'] > oversold, 'cci_bullish_signal'] = 1
    signals.loc[signals['CCI'] < overbought, 'cci_bearish_signal'] = 1

    # Generate buy/sell signals based on overbought/oversold conditions
    signals['cci_buy_signal'] = 0
    signals['cci_sell_signal'] = 0
    signals.loc[(signals['CCI'] > overbought) & (signals['CCI'].shift(1) <= overbought), 'cci_buy_signal'] = 1
    signals.loc[(signals['CCI'] < oversold) & (signals['CCI'].shift(1) >= oversold), 'cci_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['CCI'], axis=1, inplace=True)

    return signals

#%% 
# Detrended Price Oscillator (DPO) Strategy
