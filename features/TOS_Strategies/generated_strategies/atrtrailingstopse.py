import pandas as pd
import numpy as np
import talib

def atr_trailing_stop_se_signals(stock_df, atr_period=14, atr_factor=3.0, trail_type="unmodified"):

    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']

    # Depending on the trail type, use either unmodified ATR or a modified ATR calculation
    if trail_type == "unmodified":
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
    elif trail_type == "modified":
        atr = atr * atr_factor
    else:
        raise ValueError(f"Unsupported trail type '{trail_type}'")

    # Calculate the ATR Trailing Stop value
    atr_trailing_stop = close.shift() - atr

    # Create a DataFrame for the signals
    signals = pd.DataFrame({'atr_trailing_stop': atr_trailing_stop}, index=stock_df.index)
    signals['atr_se_signal'] = 'neutral'
    signals.loc[close < signals['atr_trailing_stop'], 'atr_se_signal'] = 'short'

    # Drop the internal 'atr_trailing_stop' column, as it is not part of the output
    signals.drop(columns=['atr_trailing_stop'], inplace=True)

    return signals
