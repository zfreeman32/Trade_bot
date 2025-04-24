
import pandas as pd
import numpy as np

# Choppiness Index Strategy
def choppiness_index_signals(stock_df, window=14, choppy_threshold=61.8, trending_threshold=38.2):
    """
    Generate trading signals based on the Choppiness Index.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with a DateTime index and 'Close' column.
    window (int): The lookback period for the Choppiness Index calculation.
    choppy_threshold (float): Crossover value to indicate a choppy market.
    trending_threshold (float): Crossover value to indicate a trending market.

    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the high and low over the specified window
    high = stock_df['Close'].rolling(window=window).max()
    low = stock_df['Close'].rolling(window=window).min()
    
    # Calculate the Choppiness Index
    choppiness_index = 100 * np.log10((high - low) / (high - stock_df['Close'].rolling(window=window).mean())) / np.log10(window)
    signals['Choppiness Index'] = choppiness_index
    
    # Generate signals based on the Choppiness Index
    signals['chop_signal'] = 'neutral'
    signals.loc[signals['Choppiness Index'] > choppy_threshold, 'chop_signal'] = 'short'  # Choppy Market
    signals.loc[signals['Choppiness Index'] < trending_threshold, 'chop_signal'] = 'long'  # Trending Market
    
    return signals[['chop_signal']]
