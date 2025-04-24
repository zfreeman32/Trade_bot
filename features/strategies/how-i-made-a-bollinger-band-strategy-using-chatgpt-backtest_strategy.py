
import pandas as pd
import numpy as np

# Bollinger Bands Mean Reversion Strategy
def bollinger_band_mean_reversion_signals(stock_df, window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    rolling_mean = stock_df['Close'].rolling(window=window).mean()
    rolling_std = stock_df['Close'].rolling(window=window).std()
    
    signals['Upper Band'] = rolling_mean + (rolling_std * num_std_dev)
    signals['Lower Band'] = rolling_mean - (rolling_std * num_std_dev)
    
    signals['signal'] = 'neutral'
    
    signals.loc[stock_df['Close'] > signals['Upper Band'], 'signal'] = 'short'
    signals.loc[stock_df['Close'] < signals['Lower Band'], 'signal'] = 'long'
    
    return signals[['signal']]
