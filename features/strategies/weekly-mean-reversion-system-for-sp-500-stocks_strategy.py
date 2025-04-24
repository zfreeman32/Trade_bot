
import pandas as pd
import numpy as np

# Weekly Mean Reversion Strategy for S&P 500 Stocks
def weekly_mean_reversion_signals(stock_df, window=5, threshold=0.05):
    """
    Generates trading signals based on a weekly mean reversion strategy for S&P 500 stocks.
    
    Parameters:
        stock_df (DataFrame): DataFrame containing stock price data with a DateTime index and a 'Close' column.
        window (int): Number of weeks to calculate the mean price.
        threshold (float): Percentage threshold to determine overbought or oversold conditions.
        
    Returns:
        DataFrame: A DataFrame with trading signals ('long', 'short', 'neutral').
    """
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    
    # Calculate the weekly mean
    signals['Mean'] = signals['Close'].rolling(window=window).mean()
    signals['Mean'] = signals['Mean'].shift(1)  # Use the mean of the previous week for signals
    
    # Calculate the price deviation from the mean
    signals['Deviation'] = (signals['Close'] - signals['Mean']) / signals['Mean']
    
    # Generate signals based on the deviation
    signals['Signal'] = 'neutral'
    signals.loc[signals['Deviation'] < -threshold, 'Signal'] = 'long'  # Buy when price drops significantly below mean
    signals.loc[signals['Deviation'] > threshold, 'Signal'] = 'short'  # Sell when price rises significantly above mean
    
    return signals[['Signal']]
