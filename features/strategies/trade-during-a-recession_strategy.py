
import pandas as pd
import numpy as np
import datetime as dt

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    """
    Generates trading signals based on the membership plans trading strategy.
    This strategy evaluates quantifiable market patterns to generate long, short, or neutral signals.
    Monthly strategies are published, except in July and December.
    """

    # Initialize signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    
    # Convert the index to datetime if it's not
    if not pd.api.types.is_datetime64_any_dtype(stock_df.index):
        stock_df.index = pd.to_datetime(stock_df.index)

    # Define a simple trading rule: 
    # Buy when price crosses above the 20-day moving average,
    # Sell when price crosses below the 20-day moving average
    moving_average = stock_df['Close'].rolling(window=20).mean()
    
    signals['moving_average'] = moving_average
    
    # Generate long and short signals
    signals.loc[(stock_df['Close'] > moving_average) & (stock_df['Close'].shift(1) <= moving_average.shift(1)), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < moving_average) & (stock_df['Close'].shift(1) >= moving_average.shift(1)), 'signal'] = 'short'

    # Clean up the DataFrame
    signals.drop(['moving_average'], axis=1, inplace=True)

    return signals
