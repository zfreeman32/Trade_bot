
import pandas as pd

# Membership Strategy Trading Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    
    # Generate long signal based on certain conditions
    signals.loc[(
        stock_df['Close'] > stock_df['Close'].rolling(window=5).mean()  # If price is above 5-day moving average
        ) & (
        stock_df['Volume'] > stock_df['Volume'].rolling(window=5).mean()  # and volume is above 5-day average
    ), 'membership_signal'] = 'long'
    
    # Generate short signal based on certain conditions
    signals.loc[(
        stock_df['Close'] < stock_df['Close'].rolling(window=5).mean()  # If price is below 5-day moving average
        ) & (
        stock_df['Volume'] < stock_df['Volume'].rolling(window=5).mean()  # and volume is below 5-day average
    ), 'membership_signal'] = 'short'
    
    return signals
