
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df, bonus_strategies_count=15, monthly_strategies_count=10):
    """
    Generate trading signals based on the quantified trading strategies membership plans.
    
    Param:
    stock_df: DataFrame containing stock data with 'Close' prices
    bonus_strategies_count: Number of strategies chosen as a signup bonus
    monthly_strategies_count: Number of backtested strategies received monthly
    
    Returns:
    signals: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Example of using different strategy counts as parameters for decision making
    if bonus_strategies_count > 10:
        signals['signal'] = 'long'  # More strategies potentially indicate a bullish sentiment
    else:
        signals['signal'] = 'short'  # Fewer strategies may suggest caution
    
    # Adds a neutral condition based on ongoing market conditions suggested by the number of monthly strategies
    if monthly_strategies_count < 5:
        signals['signal'] = 'neutral'
    
    return signals
