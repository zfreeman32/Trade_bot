
import pandas as pd
import numpy as np

# Membership Plans Strategy
def membership_plans_signals(stock_df, membership_type='Gold'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simple conditions for generating signals based on membership type
    if membership_type == 'Platinum':
        signals['signal'] = np.where(stock_df['Close'] > stock_df['Open'], 'long', 'short')
    elif membership_type == 'Gold':
        signals['signal'] = np.where(stock_df['Close'] < stock_df['Open'], 'short', 'long')
    else:
        signals['signal'] = 'neutral'
    
    return signals
