import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Day of the week
def day_of_the_week(df):
    # Make a copy of the dataframe to avoid modifying the original data
    df_copy = df.copy()
    
    # Check if the index is a datetime index
    if isinstance(df_copy.index, pd.DatetimeIndex):
        # Pandas dayofweek: Monday=0, Sunday=6; add 1 to map to 1-7
        day_numbers = df_copy.index.dayofweek + 1
    # Alternatively, if you have a 'Date' column:
    elif 'Date' in df_copy.columns:
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        day_numbers = df_copy['Date'].dt.dayofweek + 1
    else:
        raise ValueError("No datetime index or 'Date' column found in the DataFrame.")
    
    # Create a new DataFrame with the computed day of week numbers
    days_df = pd.DataFrame({'day_of_week': day_numbers}, index=df_copy.index)
    return days_df

