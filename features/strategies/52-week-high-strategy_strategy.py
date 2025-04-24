
import pandas as pd

# 52-Week High Trading Strategy
def fifty_two_week_high_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the 52-week high
    stock_df['52_week_high'] = stock_df['Close'].rolling(window=252).max()
    
    # Create signals based on 52-week high proximity
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] >= 0.95 * stock_df['52_week_high'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] < 0.90 * stock_df['52_week_high'], 'signal'] = 'short'

    return signals[['signal']]
