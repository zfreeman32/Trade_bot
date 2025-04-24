
import pandas as pd
import numpy as np

# Turn of the Month Trading Strategy
def turn_of_the_month_signals(stock_df):
    # Create an empty DataFrame to hold signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    
    # Ensure the dataframe is sorted by date
    stock_df = stock_df.sort_index()
    
    # Get the trading days
    trading_days = stock_df.index[stock_df.index.to_series().dt.dayofweek < 5]  # Weekdays only
    
    for i in range(len(trading_days)):
        if i < 4:
            continue  # Skip the first 4 days as we won't be able to evaluate the 5th last day
            
        current_date = trading_days[i]
        if current_date.is_month_end and (trading_days[i - 4] == current_date):
            buy_date = current_date
            exit_date = current_date + pd.DateOffset(days=7)  # Exit on the 3rd trading day of next month
            
            # Mark the signal
            signals.loc[buy_date, 'signal'] = 'long'
            
            # Ensure we only exit on the allowed exit date and it's a weekday
            if exit_date in stock_df.index:
                signals.loc[exit_date, 'signal'] = 'exit'
    
    # Forward fill the positions to represent ongoing signals until we exit
    signals['signal'] = signals['signal'].replace('exit', 'neutral').ffill()
    
    return signals
