
import pandas as pd
import numpy as np

# Monthly Trading Strategy based on Quantified Patterns
def quantified_trading_signals(stock_df):
    """
    Generate trading signals based on quantified trading patterns and anomalies.
    
    Parameters:
    stock_df (pd.DataFrame): Dataframe containing stock price data with a 'Close' column.

    Returns:
    signals (pd.DataFrame): DataFrame containing trading signals: 'long', 'short', 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the monthly return
    stock_df['Monthly_Return'] = stock_df['Close'].pct_change(periods=30)
    
    # Define conditions for signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Monthly_Return'] > 0), 'signal'] = 'long'  # Long if monthly return is positive
    signals.loc[(stock_df['Monthly_Return'] < 0), 'signal'] = 'short'  # Short if monthly return is negative

    return signals[['signal']]
