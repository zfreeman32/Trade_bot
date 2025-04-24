
import pandas as pd
import numpy as np

# Membership Trading Strategy Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Generating a membership signal based on simplistic entry and exit rules
    # Here we are assuming a basic approach for the purpose of demonstration.
    # In practice, this would require defined and tested conditions.

    # Conditions for membership strategy signals
    signals['membership_signal'] = 'neutral'

    # Example logic: 
    # - Buy when the price increases by more than 2% over the last 1 week
    # - Sell when the price decreases by more than 2% over the last 1 week
    
    signals['price_change'] = stock_df['Close'].pct_change(periods=5)  # Change over last week
    signals.loc[signals['price_change'] > 0.02, 'membership_signal'] = 'long'
    signals.loc[signals['price_change'] < -0.02, 'membership_signal'] = 'short'

    # Cleanup temporary columns
    signals.drop(['price_change'], axis=1, inplace=True)
    
    return signals
