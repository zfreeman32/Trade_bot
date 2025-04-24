
import pandas as pd
import numpy as np
from ta import trend, momentum

# GBP/NZD Trading Strategy
def gbpnzd_trading_signals(forex_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = forex_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = forex_df['Close'].rolling(window=long_window).mean()
    
    # Initiate signals as neutral
    signals['signal'] = 'neutral'
    
    # Generate long signals
    signals.loc[(signals['short_mavg'] > signals['long_mavg']), 'signal'] = 'long'
    
    # Generate short signals
    signals.loc[(signals['short_mavg'] < signals['long_mavg']), 'signal'] = 'short'
    
    return signals[['signal']]
