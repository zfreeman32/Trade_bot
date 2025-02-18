import pandas as pd
import numpy as np

def key_rev_lx_signals(stock_df, length=5):
    """
    Generates KeyRevLX strategy signals.

    Parameters:
      stock_df (pd.DataFrame): stock dataset with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
      length: The number of preceding bars whose High prices are compared to the current High.

    Returns:
      pd.DataFrame: signals with long exit   
    """

    # Create DataFrame for signals.
    signals = pd.DataFrame(index=stock_df.index)
    signals['key_rev_lx_signals'] = 0

    # Check the condition for each row
    for i in range(length, len(stock_df)):
        if stock_df['High'].iloc[i] > stock_df['High'].iloc[i-length:i].max() and \
           stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i-1]:
            signals.iloc[i, signals.columns.get_loc('key_rev_lx_signals')] = -1  # Use .iloc[] for safe modification

    return signals

