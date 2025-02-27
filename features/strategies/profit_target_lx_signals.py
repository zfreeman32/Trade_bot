import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from profittargetlx.py
def profit_target_lx_signals(stock_df, target=0.01, offset_type="percent"):
    signals = pd.DataFrame(index=stock_df.index)
    signals['profit_target_sell_signal'] = 0

    if offset_type=="percent":
        exit_price = stock_df['Close'] * (1 + target)
    elif offset_type=="tick":
        exit_price = stock_df['Close'] + (stock_df['Close'].diff() * target)
    elif offset_type=="value":
        exit_price = stock_df['Close'] + target
    else:
        return "Invalid offset type. Please use 'percent', 'tick' or 'value'."

    signals.loc[stock_df['Close'].shift(-1) >= exit_price, 'profit_target_sell_signal'] = 1

    return signals
