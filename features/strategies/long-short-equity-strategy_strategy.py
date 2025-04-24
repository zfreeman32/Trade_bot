
import pandas as pd

# Long Short Equity Strategy
def long_short_equity_signals(stock_df, long_criteria_func, short_criteria_func):
    """
    This function generates long and short trading signals based on given criteria functions 
    for long and short positions respectively.
    
    Parameters:
    - stock_df (DataFrame): DataFrame containing stock data with a 'Close' column.
    - long_criteria_func (function): Function that takes stock_df and returns a boolean Series for long positions.
    - short_criteria_func (function): Function that takes stock_df and returns a boolean Series for short positions.
    
    Returns:
    - DataFrame: DataFrame with signals 'long', 'short', or 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Generate long signals
    long_signals = long_criteria_func(stock_df)
    signals.loc[long_signals, 'signal'] = 'long'
    
    # Generate short signals
    short_signals = short_criteria_func(stock_df)
    signals.loc[short_signals, 'signal'] = 'short'
    
    return signals

# Example Criteria Functions
# These would normally be replaced with actual logic to identify long and short positions
def example_long_criteria(stock_df):
    return stock_df['Close'].pct_change().rolling(window=5).mean() > 0.01  # Arbitrary example

def example_short_criteria(stock_df):
    return stock_df['Close'].pct_change().rolling(window=5).mean() < -0.01  # Arbitrary example
