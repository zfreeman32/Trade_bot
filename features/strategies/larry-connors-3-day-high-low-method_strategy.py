
import pandas as pd

# Larry Connors' 3 Day High/Low Method Trading Strategy
def connors_3_day_high_low_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate 3-day high and 3-day low
    stock_df['3_day_high'] = stock_df['Close'].rolling(window=3).max()
    stock_df['3_day_low'] = stock_df['Close'].rolling(window=3).min()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] < stock_df['3_day_low'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] > stock_df['3_day_high'], 'signal'] = 'short'
    
    return signals[['signal']]
