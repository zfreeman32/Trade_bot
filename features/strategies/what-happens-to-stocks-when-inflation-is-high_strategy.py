
import pandas as pd
import numpy as np

# High Inflation Trading Strategy
def inflation_trading_signals(stock_df, inflation_threshold=3.0):
    """
    Generate trading signals based on high inflation and its effect on stocks.
    
    Parameters:
        stock_df (DataFrame): DataFrame containing stock price data with a 'Close' column.
        inflation_threshold (float): The threshold for high inflation (in percentage).
        
    Returns:
        signals (DataFrame): DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Creating a dummy inflation series for testing purposes (replace with actual inflation data)
    # The inflation data should be in percentage
    inflation_data = pd.Series(np.random.uniform(1.0, 5.0, size=len(stock_df)), index=stock_df.index)
    
    # Determine high inflation periods
    high_inflation = inflation_data > inflation_threshold
    
    # Generate signals based on high inflation conditions
    signals['signal'] = 'neutral'
    signals.loc[high_inflation, 'signal'] = 'short'  # Short during high inflation
    signals.loc[~high_inflation, 'signal'] = 'long'  # Long during low inflation
    
    return signals
