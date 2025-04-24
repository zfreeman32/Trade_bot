
import pandas as pd
import numpy as np
from ta import trend

# EURNZD Forex Trading Strategy
def eurnzd_forex_strategy(price_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=price_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = price_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = price_df['Close'].rolling(window=long_window).mean()
    
    # Initialize the 'signal' column
    signals['signal'] = 'neutral'
    
    # Generate trading signals
    signals.loc[(signals['short_mavg'] > signals['long_mavg']), 'signal'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']), 'signal'] = 'short'
    
    # Clean up the DataFrame by dropping the moving average columns
    signals.drop(['short_mavg', 'long_mavg'], axis=1, inplace=True)
    
    return signals
