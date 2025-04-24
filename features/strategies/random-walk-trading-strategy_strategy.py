
import pandas as pd
import numpy as np

# Random Walk Trading Strategy
def random_walk_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    np.random.seed(42)  # For reproducibility
    
    # Generate random numbers for trading decisions
    random_numbers = np.random.rand(len(stock_df))
    
    # Define thresholds for trading signals
    buy_threshold = 0.5
    sell_threshold = 0.5
    
    # Create signals based on random numbers
    signals['random_signal'] = 'neutral'
    signals.loc[random_numbers < buy_threshold, 'random_signal'] = 'long'
    signals.loc[random_numbers > sell_threshold, 'random_signal'] = 'short'
    
    return signals[['random_signal']]
