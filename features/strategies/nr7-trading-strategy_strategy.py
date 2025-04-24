
import pandas as pd

# NR7 Trading Strategy
def nr7_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['NR7'] = (stock_df['High'].rolling(window=7).max() - stock_df['Low'].rolling(window=7).min()).rolling(window=1).apply(lambda x: x[-1])
    signals['Current_Range'] = stock_df['High'] - stock_df['Low']
    
    # Define conditions for NR7
    signals['is_NR7'] = (signals['Current_Range'] < signals['NR7'].rolling(window=6).min())
    
    # Signal generation
    signals['signal'] = 'neutral'
    signals.loc[signals['is_NR7'], 'signal'] = 'long'
    
    # Remove extra columns
    signals.drop(['NR7', 'Current_Range', 'is_NR7'], axis=1, inplace=True)
    
    return signals
