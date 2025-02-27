import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from halloween.py
def halloween_strategy(data: pd.DataFrame, sma_length: int = 30):
    signals = pd.DataFrame(index=data.index)
    
    
    # Create SMA Indicator
    sma = trend.SMAIndicator(data["Close"], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Create a signal column and initialize it to Hold.
    signals['halloween_strategy_buy_signal'] = 0
    signals['halloween_strategy_sell_signal'] = 0
    
    # Generate Long Entry signal
    signals.loc[(signals.index.month == 10) & (signals.index.day == 1) & (data['Close'] > signals['SMA']), 'halloween_strategy_buy_signal'] = 1
    
    # Generate Long Exit Signal
    signals.loc[(signals.index.month == 5) & (signals.index.day == 1), 'halloween_strategy_sell_signal'] = 1
    signals.drop(['SMA'], axis=1, inplace=True)
    return signals
