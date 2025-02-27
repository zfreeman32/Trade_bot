import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from momentumle.py
def momentumle_signals(stock_df, length=12, price_scale=100):
    # Initialize the signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Momentum
    mom = momentum.roc(stock_df['Close'], window=length)
    
    # Create buy and sell signal columns
    signals['momentumle_buy_signal'] = np.where((mom > 0) & (mom.shift(1) <= 0), 1, 0)
    
    # Calculate signal price level
    signal_price_level = stock_df['High'] + (1 / price_scale)
    
    # Generate buy signals if price level condition is met
    signals['momentumle_buy_signal'] = np.where(stock_df['Open'].shift(-1) > signal_price_level, 1, signals['momentumle_buy_signal'])
        
    return signals
