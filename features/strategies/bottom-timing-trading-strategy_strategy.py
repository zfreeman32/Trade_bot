
import pandas as pd

# Membership Trading Strategy Signals
def membership_trading_signals(membership_df):
    signals = pd.DataFrame(index=membership_df.index)
    
    # Example criteria for generating signals; this can be adjusted as needed
    signals['strategy_score'] = membership_df['Platinum_savings'] + membership_df['Gold_savings']
    
    # Generate signals based on the calculated strategy score
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['strategy_score'] > 0, 'trading_signal'] = 'long'  # Long when there are savings
    signals.loc[signals['strategy_score'] < 0, 'trading_signal'] = 'short'  # Short when no savings or negative
    
    return signals[['trading_signal']]
