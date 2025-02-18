import pandas as pd
import numpy as np

def profit_target_SX(df, target=0.75, offset_type='value', tick_size=0.01):
    signals = pd.DataFrame(index=df.index)
    signals['profit_target_SX_signals'] = 0
    if offset_type == 'value':
        signals['profit_target_SX_signals'] = np.where(df['Close'].diff() <= -target, 'Short Exit', signals['profit_target_SX_signals'])
    elif offset_type == 'tick':
        signals['profit_target_SX_signals'] = np.where(df['Close'].diff() <= -(target * tick_size), 'Short Exit', signals['profit_target_SX_signals'])
    elif offset_type == 'percent':
        signals['profit_target_SX_signals'] = np.where(df['Close'].pct_change() <= -target/100, 'Short Exit', signals['profit_target_SX_signals'])  
    return signals

