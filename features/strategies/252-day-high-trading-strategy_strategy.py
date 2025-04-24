
import pandas as pd

# 252-Day High Trading Strategy
def high_252_day_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate the rolling maximum for the last 252 days
    signals['252_day_high'] = stock_df['Close'].rolling(window=252).max()
    
    # Generate signals based on the condition that today's close is a 252-day high
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] >= signals['252_day_high'], 'signal'] = 'long'
    
    # No short signal defined in the strategy, it holds long positions at the 252-day high
    return signals[['signal']]
