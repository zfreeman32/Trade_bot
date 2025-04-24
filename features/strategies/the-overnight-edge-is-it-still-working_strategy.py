
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Quantified Strategies Monthly Trading Strategy
def quantified_strategies_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the average price over the last 20 periods
    signals['average_price'] = stock_df['Close'].rolling(window=20).mean()
    
    # Calculate the standard deviation over the last 20 periods
    signals['std_dev'] = stock_df['Close'].rolling(window=20).std()
    
    # Define long and short entry points based on the average price and standard deviation
    signals['long_entry'] = signals['average_price'] + signals['std_dev']
    signals['short_entry'] = signals['average_price'] - signals['std_dev']
    
    # Initialize the signal column
    signals['strategy_signal'] = 'neutral'
    
    # Generate long signals
    signals.loc[stock_df['Close'] > signals['long_entry'], 'strategy_signal'] = 'long'
    
    # Generate short signals
    signals.loc[stock_df['Close'] < signals['short_entry'], 'strategy_signal'] = 'short'
    
    # Clean up the signals DataFrame
    signals.drop(['average_price', 'std_dev', 'long_entry', 'short_entry'], axis=1, inplace=True)
    
    return signals
