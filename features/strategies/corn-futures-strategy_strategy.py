
import pandas as pd
import numpy as np
from ta import momentum

# Mean Reversion Strategy
def mean_reversion_signals(stock_df, window=20, entry_zscore=1.0, exit_zscore=0.0):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the rolling mean and standard deviation
    rolling_mean = stock_df['Close'].rolling(window=window).mean()
    rolling_std = stock_df['Close'].rolling(window=window).std()
    
    # Calculate z-scores
    signals['z_score'] = (stock_df['Close'] - rolling_mean) / rolling_std
    
    signals['signal'] = 'neutral'
    signals.loc[signals['z_score'] <= -entry_zscore, 'signal'] = 'long'  # Buy signal
    signals.loc[signals['z_score'] >= entry_zscore, 'signal'] = 'short'  # Sell signal
    signals.loc[signals['z_score'] <= exit_zscore, 'signal'] = 'neutral' # Exit the position

    return signals[['signal']]
