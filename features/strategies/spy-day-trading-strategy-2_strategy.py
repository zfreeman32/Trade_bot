
import pandas as pd
import numpy as np
from ta import momentum

# Candlestick Pattern Strategy
def candlestick_pattern_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'

    # Example of a simple candlestick pattern: Bullish Engulfing
    for i in range(1, len(stock_df)):
        # Check for Bullish Engulfing Pattern
        if (stock_df['Close'].iloc[i] > stock_df['Open'].iloc[i] and
            stock_df['Close'].iloc[i-1] < stock_df['Open'].iloc[i-1] and
            stock_df['Open'].iloc[i] < stock_df['Close'].iloc[i-1] and
            stock_df['Close'].iloc[i] > stock_df['Open'].iloc[i-1]):
            signals['signal'].iloc[i] = 'long'

        # Check for Bearish Engulfing Pattern
        elif (stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i] and
              stock_df['Close'].iloc[i-1] > stock_df['Open'].iloc[i-1] and
              stock_df['Open'].iloc[i] > stock_df['Close'].iloc[i-1] and
              stock_df['Close'].iloc[i] < stock_df['Open'].iloc[i-1]):
            signals['signal'].iloc[i] = 'short'

    return signals
