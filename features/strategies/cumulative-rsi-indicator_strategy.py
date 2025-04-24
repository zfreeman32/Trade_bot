
import pandas as pd
import numpy as np
from ta import momentum

# Cumulative RSI Strategy
def cumulative_rsi_signals(stock_df, rsi_window=14, cumulative_window=2, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Standard RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_window)
    stock_df['RSI'] = rsi.rsi()
    
    # Calculate Cumulative RSI
    stock_df['Cumulative_RSI'] = stock_df['RSI'].rolling(window=cumulative_window).sum()
    
    # Generate signals based on Cumulative RSI
    signals['cumulative_rsi_signal'] = 'neutral'
    # Long signal
    signals.loc[(stock_df['Cumulative_RSI'] < oversold), 'cumulative_rsi_signal'] = 'long'
    # Short signal
    signals.loc[(stock_df['Cumulative_RSI'] > overbought), 'cumulative_rsi_signal'] = 'short'
    
    return signals[['cumulative_rsi_signal']]
