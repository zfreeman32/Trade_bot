import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from stoplosslx.py
def stop_loss_lx_signals(stock_df, offset_type="percent", stop=0.75):

    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_sell_signal'] = 0
            
    if offset_type.lower() == "value":
        stop_loss_price = stock_df['Close'].shift(1) - stop
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    if offset_type.lower() == "tick":
        stop_loss_price = stock_df['Close'].shift(1) - (stop * stock_df['TickSize'])
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    if offset_type.lower() == "percent":
        stop_loss_price = stock_df['Close'].shift(1) - (stock_df['Close'].shift(1) * stop/100)
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    return signals
