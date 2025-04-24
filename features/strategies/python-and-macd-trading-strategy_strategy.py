
import pandas as pd
import numpy as np
import yfinance as yf
from ta.trend import MACD

# MACD Trading Strategy
def macd_signals(stock_df):
    # Calculate MACD
    macd = MACD(stock_df['Close'])
    stock_df['MACD'] = macd.macd()
    stock_df['Signal'] = macd.macd_signal()
    stock_df['Histogram'] = macd.macd_diff()
    
    # Generate signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['macd_signal'] = 'neutral'
    signals.loc[stock_df['Histogram'] > 0, 'macd_signal'] = 'long'  # Buy Signal
    signals.loc[stock_df['Histogram'] < 0, 'macd_signal'] = 'short'  # Sell Signal
    
    return signals
