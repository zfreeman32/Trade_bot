
import pandas as pd

# Membership Based Trading Strategy
def membership_based_trading_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Simplified logic for membership-based trading signals
    signals['membership_signal'] = 'neutral'
    
    # Assume we have a simple rule: 
    # Buy signal when a closing price is greater than the moving average of the last 10 days
    # Sell signal when the closing price is less than the moving average of the last 10 days
    moving_average = stock_df['Close'].rolling(window=10).mean()
    
    signals.loc[stock_df['Close'] > moving_average, 'membership_signal'] = 'long'
    signals.loc[stock_df['Close'] < moving_average, 'membership_signal'] = 'short'
    
    return signals
