
import pandas as pd
import numpy as np

# Kairi Relative Index (KRI) Strategy
def kairi_relative_index_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Simple Moving Average (SMA)
    sma = stock_df['Close'].rolling(window=window).mean()
    
    # Calculate Kairi Relative Index (KRI)
    kri = (stock_df['Close'] - sma) / sma * 100
    signals['KRI'] = kri

    signals['kri_signal'] = 'neutral'
    signals.loc[(signals['KRI'] > 10) & (signals['KRI'].shift(1) <= 10), 'kri_signal'] = 'short'
    signals.loc[(signals['KRI'] < -10) & (signals['KRI'].shift(1) >= -10), 'kri_signal'] = 'long'

    signals.drop(['KRI'], axis=1, inplace=True)
    return signals
