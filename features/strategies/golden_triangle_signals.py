import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Functions from goldentrianglele.py
def golden_triangle_signals(stock_df, average_length=50, confirm_length=20, volume_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    sma_long = trend.SMAIndicator(stock_df['Close'], average_length)
    sma_short = trend.SMAIndicator(stock_df['Close'], confirm_length)
    
    # Identify initial uptrend condition
    uptrend = stock_df['Close'] > sma_long.sma_indicator()
    
    # Identify pivot points as local maxima
    pivot = ((stock_df['Close'] > stock_df['Close'].shift()) &
             (stock_df['Close'] > stock_df['Close'].shift(-1)))
    
    # Identify price drop condition
    price_drop = stock_df['Close'] < sma_long.sma_indicator()
    
    # Define initial triangle setup condition
    triangle_setup = (uptrend & pivot & price_drop).shift().fillna(False)
    
    # Price and volume confirmation
    price_confirm = stock_df['Close'] > sma_short.sma_indicator()
    volume_confirm = stock_df['Volume'] > stock_df['Volume'].rolling(volume_length).max()
    triangle_confirm = (price_confirm & volume_confirm).shift().fillna(False)
    
    # Generate buy signals
    signals['golden_triangle_buy_signal'] = np.where(triangle_setup & triangle_confirm, 1, 0)
    
    return signals
