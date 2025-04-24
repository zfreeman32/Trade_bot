
import pandas as pd
import numpy as np

# Quantified Patterns and Anomalies Strategy
def quantified_patterns_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'

    # Define a moving average for the strategy
    short_window = 20
    long_window = 50

    # Calculate moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()

    # Generate signals
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 'long'
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = 'short'

    # Clean up the DataFrame
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    return signals
