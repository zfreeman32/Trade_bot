
import pandas as pd

# Advance-Decline Indicator (A/D) Strategy
def ad_indicator_signals(stock_df):
    # Calculate the daily advances and declines
    stock_df['Prev Close'] = stock_df['Close'].shift(1)
    stock_df['Advancing'] = (stock_df['Close'] > stock_df['Prev Close']).astype(int)
    stock_df['Declining'] = (stock_df['Close'] < stock_df['Prev Close']).astype(int)
    
    # Cumulative sum for advances and declines
    stock_df['Cumulative Advance'] = stock_df['Advancing'].cumsum()
    stock_df['Cumulative Decline'] = stock_df['Declining'].cumsum()
    
    # Calculate A/D line
    stock_df['AD Line'] = stock_df['Cumulative Advance'] - stock_df['Cumulative Decline']
    
    # Generate trading signals based on A/D line divergence
    signals = pd.DataFrame(index=stock_df.index)
    signals['AD Line'] = stock_df['AD Line']
    signals['Signal'] = 'neutral'

    signals.loc[(signals['AD Line'] > signals['AD Line'].shift(1)) & (stock_df['Close'] > stock_df['Close'].shift(1)), 'Signal'] = 'long'
    signals.loc[(signals['AD Line'] < signals['AD Line'].shift(1)) & (stock_df['Close'] < stock_df['Close'].shift(1)), 'Signal'] = 'short'
    
    return signals[['AD Line', 'Signal']]

