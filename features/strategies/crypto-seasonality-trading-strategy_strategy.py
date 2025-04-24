
import pandas as pd

# Membership Strategy Trading Signals
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Generate signals based on membership plans
    # For simplicity, we will use a basic approach and assume we have two conditions:
    # Condition 1: If the Close price is above the 20-day moving average, we consider a 'long' signal.
    # Condition 2: If the Close price is below the 20-day moving average, we consider a 'short' signal.
    
    signals['Moving_Average'] = stock_df['Close'].rolling(window=20).mean()
    signals['membership_signal'] = 'neutral'
    
    # Long signal
    signals.loc[stock_df['Close'] > signals['Moving_Average'], 'membership_signal'] = 'long'
    
    # Short signal
    signals.loc[stock_df['Close'] < signals['Moving_Average'], 'membership_signal'] = 'short'
    
    return signals[['membership_signal']]
