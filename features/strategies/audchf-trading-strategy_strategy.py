
import pandas as pd
import numpy as np

# AUDCHF Trading Strategy
def audchf_signals(audchf_df, threshold=0.001):
    """Generate trading signals for AUDCHF based on price changes.
    
    Args:
        audchf_df (pd.DataFrame): DataFrame containing at least 'Close' price data.
        threshold (float): Price change threshold to determine buy/sell signals.

    Returns:
        pd.DataFrame: A DataFrame with the trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=audchf_df.index)
    signals['Close'] = audchf_df['Close']
    signals['Price Change'] = signals['Close'].pct_change()

    # Initialize 'signal' column
    signals['signal'] = 'neutral'

    # Generate signals based on price change
    signals.loc[signals['Price Change'] > threshold, 'signal'] = 'long'   # Buy signal
    signals.loc[signals['Price Change'] < -threshold, 'signal'] = 'short'  # Sell signal

    return signals[['Close', 'signal']]
