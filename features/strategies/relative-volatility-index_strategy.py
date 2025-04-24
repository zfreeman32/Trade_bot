
import pandas as pd
import numpy as np

# Relative Volatility Index (RVI) Strategy
def rvi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate the standard deviation of high and low prices
    high_std = stock_df['High'].rolling(window=window).std()
    low_std = stock_df['Low'].rolling(window=window).std()
    
    # Calculate the RVI
    rvi = (high_std - low_std) / (high_std + low_std)
    signals['RVI'] = rvi

    # Determine signals based on RVI
    signals['rvi_signal'] = 'neutral'
    signals.loc[(signals['RVI'] > 0) & (signals['RVI'].shift(1) <= 0), 'rvi_signal'] = 'long'
    signals.loc[(signals['RVI'] < 0) & (signals['RVI'].shift(1) >= 0), 'rvi_signal'] = 'short'
    
    signals.drop(['RVI'], axis=1, inplace=True)
    return signals
