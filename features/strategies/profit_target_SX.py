import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from profittargetsx.py
def profit_target_SX(df, target=0.75, offset_type='value', tick_size=0.01):
    signals = pd.DataFrame(index=df.index)
    signals['profit_target_buy_signal'] = 0
    if offset_type == 'value':
        signals['profit_target_buy_signal'] = np.where(df['Close'].diff() <= -target, 'Short Exit', signals['profit_target_buy_signal'])
    elif offset_type == 'tick':
        signals['profit_target_buy_signal'] = np.where(df['Close'].diff() <= -(target * tick_size), 'Short Exit', signals['profit_target_buy_signal'])
    elif offset_type == 'percent':
        signals['profit_target_buy_signal'] = np.where(df['Close'].pct_change() <= -target/100, 'Short Exit', signals['profit_target_buy_signal'])  
    return signals
