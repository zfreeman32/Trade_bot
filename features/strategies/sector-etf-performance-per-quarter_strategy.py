
import pandas as pd
import numpy as np

# Membership Strategy Signals Based on Quantified Patterns
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Example conditions for signals based on hypothetical quantified patterns
    # These conditions should be replaced with real quantifiable rules from the strategy
    signals['Price_Change'] = stock_df['Close'].pct_change()
    signals['Volatility'] = stock_df['Close'].rolling(window=14).std()

    # Long signal: price increased with low volatility
    signals['signal'] = 'neutral'
    signals.loc[(signals['Price_Change'] > 0.01) & (signals['Volatility'] < 0.02), 'signal'] = 'long'
    
    # Short signal: price decreased with low volatility
    signals.loc[(signals['Price_Change'] < -0.01) & (signals['Volatility'] < 0.02), 'signal'] = 'short'
    
    return signals[['signal']]
