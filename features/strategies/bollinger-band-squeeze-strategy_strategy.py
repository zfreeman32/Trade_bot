
import pandas as pd
import numpy as np
from ta import volatility

# Bollinger Band Squeeze Strategy
def bollinger_band_squeeze_signals(stock_df, window=25, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Bollinger Bands
    stock_df['SMA'] = stock_df['Close'].rolling(window=window).mean()
    stock_df['UpperBand'] = stock_df['SMA'] + (stock_df['Close'].rolling(window=window).std() * num_std_dev)
    stock_df['LowerBand'] = stock_df['SMA'] - (stock_df['Close'].rolling(window=window).std() * num_std_dev)

    # Identify squeeze condition
    signals['squeeze'] = (stock_df['UpperBand'] - stock_df['LowerBand']) < stock_df['LowerBand'].rolling(window=window).mean()

    # Generate trading signals based on the breakout from the squeeze
    signals['long'] = (signals['squeeze'].shift(1) == True) & (stock_df['Close'] > stock_df['UpperBand'].shift(1))
    signals['short'] = (signals['squeeze'].shift(1) == True) & (stock_df['Close'] < stock_df['LowerBand'].shift(1))
    
    # Set the signals to 'neutral' initially
    signals['position'] = 'neutral'
    signals.loc[signals['long'], 'position'] = 'long'
    signals.loc[signals['short'], 'position'] = 'short'
    
    return signals[['position']]
