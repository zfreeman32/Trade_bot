
import pandas as pd
import numpy as np

# Time Segmented Volume (TSV) Strategy
def tsv_signals(stock_df, time_segment=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate positive and negative volume
    stock_df['Positive_Volume'] = np.where(stock_df['Close'].diff() > 0, stock_df['Volume'], 0)
    stock_df['Negative_Volume'] = np.where(stock_df['Close'].diff() < 0, stock_df['Volume'], 0)
    
    positive_sum = stock_df['Positive_Volume'].rolling(window=time_segment).sum()
    negative_sum = stock_df['Negative_Volume'].rolling(window=time_segment).sum()
    
    # Time Segmented Volume calculations
    stock_df['TSV'] = positive_sum - negative_sum

    # Signal generation
    signals['tsv_signal'] = 'neutral'
    signals.loc[(stock_df['TSV'] > 0) & (stock_df['TSV'].shift(1) <= 0), 'tsv_signal'] = 'long'
    signals.loc[(stock_df['TSV'] < 0) & (stock_df['TSV'].shift(1) >= 0), 'tsv_signal'] = 'short'
    
    return signals
