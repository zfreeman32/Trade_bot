
import pandas as pd

# Santa Claus Rally Trading Strategy
def santa_claus_rally_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Define the date range for the Santa Claus Rally: December 26 to January 2
    signals['signal'] = 'neutral'
    signals.loc[(stock_df.index.day >= 26) & (stock_df.index.month == 12), 'signal'] = 'long'
    signals.loc[(stock_df.index.day <= 2) & (stock_df.index.month == 1), 'signal'] = 'long'
    
    # Mark positions outside the Santa Claus Rally period as 'neutral'
    signals.loc[~((stock_df.index.day >= 26) & (stock_df.index.month == 12) | 
                   (stock_df.index.day <= 2) & (stock_df.index.month == 1)), 'signal'] = 'neutral'
    
    return signals
