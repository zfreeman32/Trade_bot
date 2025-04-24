
import pandas as pd
import numpy as np
from ta import momentum, trend

# Trend Following Strategy
def trend_following_signals(crypto_df, window=20):
    signals = pd.DataFrame(index=crypto_df.index)
    
    # Calculate the moving average
    signals['MA'] = crypto_df['Close'].rolling(window=window).mean()
    
    # Initialize signal column
    signals['trend_signal'] = 'neutral'
    
    # Generate long and short signals
    signals.loc[(crypto_df['Close'] > signals['MA']) & (crypto_df['Close'].shift(1) <= signals['MA'].shift(1)), 'trend_signal'] = 'long'
    signals.loc[(crypto_df['Close'] < signals['MA']) & (crypto_df['Close'].shift(1) >= signals['MA'].shift(1)), 'trend_signal'] = 'short'
    
    return signals.drop(['MA'], axis=1)
