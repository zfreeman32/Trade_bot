
import pandas as pd
import numpy as np
from ta import volatility

# Bollinger Bands Trading Strategy
def bollinger_band_signals(stock_df, window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the moving average and the standard deviation
    sma = stock_df['Close'].rolling(window=window).mean()
    rolling_std = stock_df['Close'].rolling(window=window).std()
    
    # Calculate Bollinger Bands
    signals['Upper Band'] = sma + (rolling_std * num_std_dev)
    signals['Lower Band'] = sma - (rolling_std * num_std_dev)
    
    # Generate signals
    signals['Signal'] = 'neutral'
    signals.loc[(stock_df['Close'] < signals['Lower Band']), 'Signal'] = 'long'  # Buy signal
    signals.loc[(stock_df['Close'] > signals['Upper Band']), 'Signal'] = 'short'  # Sell signal
    
    return signals[['Signal']]
