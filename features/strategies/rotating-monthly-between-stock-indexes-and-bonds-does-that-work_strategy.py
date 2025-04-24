
import pandas as pd

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['membership_signal'] = 'neutral'
    
    # Monthly membership plan strategy rule example
    # Here we will assume two simple rules based on hypothetical membership addition/subtraction criteria
    signals.loc[stock_df['Close'].pct_change() > 0.01, 'membership_signal'] = 'long'   # Buy if price increased by more than 1%
    signals.loc[stock_df['Close'].pct_change() < -0.01, 'membership_signal'] = 'short'  # Sell if price decreased by more than 1%

    return signals
