import pandas as pd
import numpy as np

def stop_loss_sx_signals(stock_df, offset_type='percent', stop=0.75):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_sx_signals'] = 0

    if offset_type.lower() == "value":
        signals.loc[(stock_df['Close'] - stock_df['Close'].shift() > stop), 'stop_loss_sx_signals'] = 1
    elif offset_type.lower() == "tick":
        tick_sizes = (stock_df['High'] - stock_df['Low']) / 2  # Assume the tick size is half the daily range
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / tick_sizes) > stop, 'stop_loss_sx_signals'] = 1
    elif offset_type.lower() == "percent":
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / stock_df['Close'].shift()) * 100 > stop, 'stop_loss_sx_signals'] = 1

    return signals

