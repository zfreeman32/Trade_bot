
import pandas as pd

# 200 Day Moving Average Trading Strategy
def dma200_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 200-day moving average
    signals['200_MA'] = stock_df['Close'].rolling(window=200).mean()
    
    # Initialize signals
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['200_MA'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['200_MA'], 'signal'] = 'short'
    
    return signals[['signal']]
