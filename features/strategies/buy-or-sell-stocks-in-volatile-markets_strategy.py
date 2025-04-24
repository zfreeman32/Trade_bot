
import pandas as pd

# Membership Plans Trading Signals Strategy
def membership_plan_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_change'] = stock_df['Close'].pct_change()
    signals['strategy_signal'] = 'neutral'
    
    # Define the conditions for long and short signals based on hypothetical strategy outcomes
    signals.loc[signals['price_change'] > 0.02, 'strategy_signal'] = 'long'  # Buy signal if price increases more than 2%
    signals.loc[signals['price_change'] < -0.02, 'strategy_signal'] = 'short'  # Sell signal if price decreases more than 2%
    
    return signals[['strategy_signal']]
