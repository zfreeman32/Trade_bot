import pandas as pd
import numpy as np
import talib

# TAC_DMI Strategy
def tac_dmi_signals(stock_df, adx_length1=14, adx_length2=14, adx_length3=14,
                  di_length1=14, di_length2=14, di_length3=14):
  
    high = stock_df['High'].values
    low = stock_df['Low'].values
    close = stock_df['Close'].values
    
    # Convert TALib outputs to Pandas Series (to allow .shift())
    tac_adx1 = pd.Series(talib.ADX(high, low, close, timeperiod=adx_length1), index=stock_df.index)
    tac_adx2 = pd.Series(talib.ADX(high, low, close, timeperiod=adx_length2), index=stock_df.index)
    tac_adx3 = pd.Series(talib.ADX(high, low, close, timeperiod=adx_length3), index=stock_df.index)
    
    tac_diplus1 = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=di_length1), index=stock_df.index)
    tac_diplus2 = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=di_length2), index=stock_df.index)
    tac_diplus3 = pd.Series(talib.PLUS_DI(high, low, close, timeperiod=di_length3), index=stock_df.index)
    
    tac_diminus1 = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=di_length1), index=stock_df.index)
    tac_diminus2 = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=di_length2), index=stock_df.index)
    tac_diminus3 = pd.Series(talib.MINUS_DI(high, low, close, timeperiod=di_length3), index=stock_df.index)
    
    # Initialize single signal column
    signals = pd.DataFrame(index=stock_df.index)
    signals['tac_dmi_signal'] = 'neutral'

    # Long Signal Condition
    long_condition = (
        (tac_diplus1 < 10) & (tac_diplus2 < 10) & (tac_diplus3 < 10) &
        (tac_adx1.shift(1) > 70) & (tac_adx2.shift(1) > 70) & (tac_adx3.shift(1) > 70) &
        (tac_adx1 < tac_adx1.shift(1)) & (tac_adx2 < tac_adx2.shift(1)) & (tac_adx3 < tac_adx3.shift(1))
    )

    # Short Signal Condition
    short_condition = (
        (tac_diminus1 < 10) & (tac_diminus2 < 10) & (tac_diminus3 < 10) &
        (tac_adx1.shift(1) > 70) & (tac_adx2.shift(1) > 70) & (tac_adx3.shift(1) > 70) &
        (tac_adx1 < tac_adx1.shift(1)) & (tac_adx2 < tac_adx2.shift(1)) & (tac_adx3 < tac_adx3.shift(1))
    )

    # Assign Signals
    signals.loc[long_condition, 'tac_dmi_signal'] = 'buy'
    signals.loc[short_condition, 'tac_dmi_signal'] = 'sell'

    return signals[['tac_dmi_signal']]

