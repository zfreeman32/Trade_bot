
import pandas as pd
import numpy as np

# Gaussian Filter Strategy
def gaussian_filter_signals(stock_df, period=14, sigma=1.0):
    """
    Generate trading signals based on the Gaussian Filter strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with a 'Close' column.
    period (int): The number of periods to calculate the Gaussian Filter.
    sigma (float): The standard deviation multiplier.

    Returns:
    pd.DataFrame: DataFrame with signals 'long', 'short', or 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Gaussian Filter
    weights = np.array([1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - (period - 1) / 2) / sigma) ** 2) for x in range(period)])
    gaussian_filter = np.convolve(stock_df['Close'], weights/weights.sum(), mode='valid')
    
    # Create signals based on Gaussian Filter trends
    signals['Filter'] = np.nan
    signals.iloc[period-1:, signals.columns.get_loc('Filter')] = gaussian_filter
    signals['signal'] = 'neutral'
    
    # Generate Buy/Sell signals
    signals.loc[(signals['Filter'] > signals['Filter'].shift(1)), 'signal'] = 'long'
    signals.loc[(signals['Filter'] < signals['Filter'].shift(1)), 'signal'] = 'short'
    
    return signals[['signal']]
