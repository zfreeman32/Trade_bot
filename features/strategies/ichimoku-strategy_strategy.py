
import pandas as pd
import numpy as np

# Ichimoku Cloud Trading Strategy
def ichimoku_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Ichimoku components
    high_9 = stock_df['High'].rolling(window=9).max()
    low_9 = stock_df['Low'].rolling(window=9).min()
    signals['Tenkan-sen'] = (high_9 + low_9) / 2
    
    high_26 = stock_df['High'].rolling(window=26).max()
    low_26 = stock_df['Low'].rolling(window=26).min()
    signals['Kijun-sen'] = (high_26 + low_26) / 2
    
    signals['Senkou Span A'] = ((signals['Tenkan-sen'] + signals['Kijun-sen']) / 2).shift(26)
    high_52 = stock_df['High'].rolling(window=52).max()
    low_52 = stock_df['Low'].rolling(window=52).min()
    signals['Senkou Span B'] = ((high_52 + low_52) / 2).shift(26)
    
    signals['Chikou Span'] = stock_df['Close'].shift(-26)
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['Tenkan-sen'] > signals['Kijun-sen']) & (signals['Chikou Span'] > signals['Close']), 'signal'] = 'long'
    signals.loc[(signals['Tenkan-sen'] < signals['Kijun-sen']) & (signals['Chikou Span'] < signals['Close']), 'signal'] = 'short'
    
    return signals[['signal']]
