
import pandas as pd
import numpy as np
from ta import momentum

# Coffee Futures Trading Strategy
def coffee_futures_signals(coffee_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=coffee_df.index)
    
    # Calculate Moving Averages
    signals['short_mavg'] = coffee_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = coffee_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate trading signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Determine when to buy or sell
    signals['positions'] = signals['signal'].diff()
    
    # Create a signal column
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
