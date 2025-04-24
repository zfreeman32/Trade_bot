
import pandas as pd
from datetime import datetime

# Membership Trading Strategy Signals
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    
    # Assume we are triggering a 'long' signal for Gold and 'short' for Platinum on the 1st of the month
    current_date = datetime.now()
    
    # Gold plan triggers on the first 10 days of the month
    if 1 <= current_date.day <= 10:
        signals['membership_signal'] = 'long'
    
    # Platinum plan triggers on the 11th to the last day of the month
    elif current_date.day > 10:
        signals['membership_signal'] = 'short'
    
    return signals
