
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Membership Plans Trading Strategy
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()
    
    # Define criteria for long and short signals based on membership plan rules
    signals['membership_signal'] = 'neutral'
    
    # Long signal when price increases by more than 2% in a day
    signals.loc[signals['price_change'] > 0.02, 'membership_signal'] = 'long'
    
    # Short signal when price decreases by more than 2% in a day
    signals.loc[signals['price_change'] < -0.02, 'membership_signal'] = 'short'
    
    # Drop temporary calculations
    signals.drop(['price_change'], axis=1, inplace=True)
    
    return signals
