
import pandas as pd
import numpy as np

# Trend Reversal Trading Strategy
def trend_reversal_signals(stock_df, lookback_period=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the rolling maximum and minimum
    rolling_max = stock_df['Close'].rolling(window=lookback_period).max()
    rolling_min = stock_df['Close'].rolling(window=lookback_period).min()

    # Create signals based on the trend reversal criteria
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > rolling_max.shift(1)), 'signal'] = 'long'  # Break above previous high
    signals.loc[(stock_df['Close'] < rolling_min.shift(1)), 'signal'] = 'short'  # Break below previous low

    return signals[['signal']]
