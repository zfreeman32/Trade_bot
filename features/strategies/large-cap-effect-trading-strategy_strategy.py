
import pandas as pd

# Membership Plans Trading Strategy
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simple example of generating signals based on membership plans strategy
    # For illustration, let's assume that the strategy checks price movements
    # on specific monthly intervals and applies a basic buying/selling logic.
    
    signals['monthly_return'] = stock_df['Close'].pct_change()
    
    # Determine signals based on the assumption of monthly entries 
    signals['signal'] = 'neutral'
    signals.loc[signals['monthly_return'] > 0.01, 'signal'] = 'long'  # Buy if return > 1%
    signals.loc[signals['monthly_return'] < -0.01, 'signal'] = 'short' # Sell if return < -1%
    
    return signals[['signal']]
