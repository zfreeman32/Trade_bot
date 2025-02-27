import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from adxbreakoutsle.py
def adx_breakouts_signals(stock_df, highest_length=15, adx_length=14, adx_level=40, offset=0.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Compute the rolling highest high and align with stock_df
    highest = stock_df['High'].rolling(window=highest_length).max()
    
    # Compute ADX and align it with stock_df
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_length).adx()
    
    # Ensure indexes are aligned
    signals['adx'] = adx
    signals['highest'] = highest
    signals['adx_breakout_buy_signal'] = 0

    # Handle potential NaNs by forward-filling
    signals.fillna(method='bfill', inplace=True)
    
    # Compute the breakout condition
    breakout_condition = (signals['adx'] > adx_level) & (stock_df['Close'] > (signals['highest'] + offset))
    
    signals.loc[breakout_condition, 'adx_breakout_buy_signal'] = 1
    
    # Drop temporary columns
    signals.drop(['adx', 'highest'], axis=1, inplace=True)

    return signals
