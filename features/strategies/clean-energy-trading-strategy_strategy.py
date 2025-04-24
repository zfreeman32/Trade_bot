
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_strategy(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()

    # Entry signal: Buy if the price change is greater than a threshold
    long_threshold = 0.01  # Example threshold for buying
    signals['signal'] = 'neutral'
    signals.loc[signals['price_change'] > long_threshold, 'signal'] = 'long'

    # Exit signal: Short if there is a price drop beyond a threshold
    short_threshold = -0.01  # Example threshold for selling
    signals.loc[signals['price_change'] < short_threshold, 'signal'] = 'short'

    return signals[['signal']]
