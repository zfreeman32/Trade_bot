
import pandas as pd
import numpy as np
from ta import momentum

# Technology Sector Trading Strategy based on ETF performance
def technology_sector_strategy(stock_df, long_window=50, short_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['long_ma'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['short_ma'] = stock_df['Close'].rolling(window=short_window).mean()
    
    # Generate trading signals
    signals['signal'] = 'neutral'
    signals.loc[signals['short_ma'] > signals['long_ma'], 'signal'] = 'long'
    signals.loc[signals['short_ma'] < signals['long_ma'], 'signal'] = 'short'
    
    # To smooth the signals and avoid indecision, forward fill the signal
    signals['signal'] = signals['signal'].ffill()
    
    return signals[['signal']]
