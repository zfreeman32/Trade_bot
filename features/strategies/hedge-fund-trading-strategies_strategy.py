
import pandas as pd
import numpy as np

# Long/Short Equity Hedge Fund Strategy
def long_short_equity_signals(stock_df, long_threshold=0.05, short_threshold=-0.05):
    """
    Generates long and short trading signals based on a simple long/short equity strategy.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with a 'Close' column.
    long_threshold (float): Threshold for generating long signals.
    short_threshold (float): Threshold for generating short signals.
    
    Returns:
    DataFrame: A DataFrame containing the trading signals.
    """
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate daily returns
    stock_df['Returns'] = stock_df['Close'].pct_change()
    
    # Initialize signal column
    signals['signal'] = 'neutral'
    
    # Generate signals based on thresholds
    signals.loc[stock_df['Returns'] > long_threshold, 'signal'] = 'long'
    signals.loc[stock_df['Returns'] < short_threshold, 'signal'] = 'short'
    
    return signals
