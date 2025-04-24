
import pandas as pd
import numpy as np

# Swing Index Strategy
def swing_index_signals(stock_df):
    """
    Generates trading signals based on the Swing Index (SI) strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with 'Open', 'High', 'Low', 'Close'.
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """

    # Calculate the Swing Index
    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']
    previous_close = close.shift(1)
    previous_high = high.shift(1)
    previous_low = low.shift(1)
    open_price = stock_df['Open']

    swing_index = (np.log(close) - np.log(previous_close)) / (np.log(previous_high) - np.log(previous_low))
    
    # Create a signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['Swing Index'] = swing_index
    signals['signal'] = 'neutral'
    
    # Generate trading signals based on the Swing Index
    signals.loc[(signals['Swing Index'] > 0) & (signals['Swing Index'].shift(1) <= 0), 'signal'] = 'long'  # Buy signal
    signals.loc[(signals['Swing Index'] < 0) & (signals['Swing Index'].shift(1) >= 0), 'signal'] = 'short'  # Sell signal
    
    return signals[['signal']]
