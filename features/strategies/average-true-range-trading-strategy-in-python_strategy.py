
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

# Average True Range (ATR) Trading Strategy
def atr_trading_signals(stock_df, atr_period=14, atr_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate ATR
    atr = AverageTrueRange(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=atr_period)
    stock_df['ATR'] = atr.average_true_range()

    # Generate signals
    signals['atr_signal'] = 'neutral'
    # Define the threshold for significant ATR
    atr_threshold = stock_df['ATR'].mean() * atr_multiplier
    
    # Long signal condition
    signals.loc[(stock_df['ATR'] > atr_threshold) & (stock_df['Close'] > stock_df['Close'].shift(1)), 'atr_signal'] = 'long'
    # Short signal condition
    signals.loc[(stock_df['ATR'] > atr_threshold) & (stock_df['Close'] < stock_df['Close'].shift(1)), 'atr_signal'] = 'short'
    
    return signals
