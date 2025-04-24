
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Candlestick Patterns Strategy
def candlestick_pattern_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the open, close, high, low for the required pattern
    signals['Open'] = stock_df['Open']
    signals['Close'] = stock_df['Close']
    signals['High'] = stock_df['High']
    signals['Low'] = stock_df['Low']
    
    # Define custom function for bullish and bearish patterns
    def identify_patterns(row):
        if row['Close'] > row['Open'] and (row['Close'] - row['Open']) > (0.01 * row['Open']):  # Bullish pattern
            return 'long'
        elif row['Close'] < row['Open'] and (row['Open'] - row['Close']) > (0.01 * row['Open']):  # Bearish pattern
            return 'short'
        else:
            return 'neutral'
    
    # Apply the pattern identification
    signals['candlestick_signal'] = signals.apply(identify_patterns, axis=1)
    return signals[['candlestick_signal']]
