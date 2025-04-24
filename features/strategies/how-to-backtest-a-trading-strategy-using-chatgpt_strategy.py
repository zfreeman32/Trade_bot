
import pandas as pd
import numpy as np

# Mean Reversion Strategy Based on Bollinger Bands
def mean_reversion_bollinger_signals(stock_df, window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Upper Band'] = stock_df['Close'].rolling(window).mean() + (stock_df['Close'].rolling(window).std() * num_std_dev)
    signals['Lower Band'] = stock_df['Close'].rolling(window).mean() - (stock_df['Close'].rolling(window).std() * num_std_dev)
    signals['Signal'] = 'neutral'
    
    signals.loc[stock_df['Close'] < signals['Lower Band'], 'Signal'] = 'long'
    signals.loc[stock_df['Close'] > signals['Upper Band'], 'Signal'] = 'short'
    
    return signals[['Signal']]
