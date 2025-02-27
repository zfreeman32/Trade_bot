import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from gapreversalle.py
def gap_reversal_signals(stock_df, gap=0.10, offset=0.50):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the gap from the previous day's low
    stock_df['PrevLow'] = stock_df['Low'].shift(1)
    stock_df['Gap'] = (stock_df['Open'] - stock_df['PrevLow']) / stock_df['PrevLow']

    # Calculate the offset from the gap
    stock_df['Offset'] = (stock_df['High'] - stock_df['Open'])
    
    # Generate signals based on condition
    signals['gap_reversal_buy_signal'] = 0
    signals.loc[(stock_df['Gap'] > gap) & (stock_df['Offset'] > offset), 'gap_reversal_buy_signal'] = 1
    
    return signals
