
import pandas as pd
import numpy as np

def stop_loss_lx_signals(stock_df, offset_type="percent", stop=0.75):

    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_lx_signal'] = 'neutral'
            
    if offset_type.lower() == "value":
        stop_loss_price = stock_df['Close'].shift(1) - stop
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    if offset_type.lower() == "tick":
        stop_loss_price = stock_df['Close'].shift(1) - (stop * stock_df['TickSize'])
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    if offset_type.lower() == "percent":
        stop_loss_price = stock_df['Close'].shift(1) - (stock_df['Close'].shift(1) * stop/100)
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    return signals
