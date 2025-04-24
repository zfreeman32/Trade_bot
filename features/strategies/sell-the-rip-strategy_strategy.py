
import pandas as pd
import numpy as np
from ta import momentum, trend

# Sell the Rip Trading Strategy
def sell_the_rip_signals(stock_df, rsi_window=14, rsi_overbought=70, bollinger_window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi.rsi()
    
    # Calculate Bollinger Bands
    middle_band = stock_df['Close'].rolling(window=bollinger_window).mean()
    std_dev = stock_df['Close'].rolling(window=bollinger_window).std()
    signals['Upper_Band'] = middle_band + (std_dev * num_std_dev)
    signals['Lower_Band'] = middle_band - (std_dev * num_std_dev)
    
    # Generate signals based on the strategy
    signals['signal'] = 'neutral'
    signals.loc[(signals['RSI'] > rsi_overbought) & (stock_df['Close'] >= signals['Upper_Band']), 'signal'] = 'short'
    
    return signals[['signal']]
