
import pandas as pd

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    # Assume the entry and exit rules based on membership plans are:
    # - Buy on the first trading day of each month (except July and December)
    # - Sell on the last trading day of each month if the position is held
    
    signals['signal'] = 'neutral'
    
    # Get the first and last trading days of each month
    monthly_groups = stock_df.resample('M')
    
    for month, group in monthly_groups:
        first_day = group.index[0]
        last_day = group.index[-1]
        
        # Buy signal
        signals.loc[first_day, 'signal'] = 'long'
        
        # Sell signal
        signals.loc[last_day, 'signal'] = 'short'
    
    # Forward fill the signals for the duration of the month
    signals['signal'] = signals['signal'].ffill()
    # Ensure no signals in July and December
    signals.loc[signals.index.month.isin([7, 12]), 'signal'] = 'neutral'
    
    return signals
