
import pandas as pd

# DAX 40 and Euro Stoxx 50 Futures Expiration Week Strategy
def expiration_week_signals(stock_df):
    """
    Generates trading signals based on the futures expiration week strategy for DAX 40 and Euro Stoxx 50.
    
    Args:
    stock_df (pd.DataFrame): DataFrame containing historical price data with a datetime index and 'Close' column.
    
    Returns:
    pd.DataFrame: DataFrame containing the signals for trading ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Extracting the day of the week and the month
    signals['Day'] = stock_df.index.dayofweek  # Monday=0, Sunday=6
    signals['Month'] = stock_df.index.month

    # Futures expiration occurs on the 3rd Friday of March, June, September, December (month numbers: 3, 6, 9, 12)
    expiration_months = [3, 6, 9, 12]
    
    # Create column for signals, initialized as 'neutral'
    signals['signal'] = 'neutral'
    
    # Apply trading strategy rules
    for date in stock_df.index:
        if (date.month in expiration_months) and (date.weekday() == 4) and (date.day >= 15 and date.day <= 21):
            # If it's a futures expiration week (3rd Friday)
            signals.loc[date, 'signal'] = 'long'  # Go long on expiration week
            
        if (date.month in expiration_months) and (date.weekday() == 0) and (date.day <= 7) and (date.day >= 1):
            # On the Monday following the expiration week
            signals.loc[date, 'signal'] = 'short'  # Go short the week after expiration
            
    return signals[['signal']]
