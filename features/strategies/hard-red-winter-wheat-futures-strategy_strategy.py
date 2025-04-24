
import pandas as pd
import numpy as np

# Hard Red Winter Wheat Trading Strategy
def kw_futures_signals(wheat_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=wheat_df.index)
    
    # Calculate the short and long moving averages
    signals['Short_MA'] = wheat_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = wheat_df['Close'].rolling(window=long_window).mean()
    
    # Initialize signals
    signals['kw_signal'] = 'neutral'
    
    # Generate signals based on moving averages
    signals.loc[(signals['Short_MA'] > signals['Long_MA']), 'kw_signal'] = 'long'
    signals.loc[(signals['Short_MA'] < signals['Long_MA']), 'kw_signal'] = 'short'
    
    # Clean up the DataFrame
    signals.drop(['Short_MA', 'Long_MA'], axis=1, inplace=True)
    
    return signals
