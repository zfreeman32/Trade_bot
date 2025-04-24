
import pandas as pd
import numpy as np

# Retirement and FIRE Risk Mitigation Strategy
def retirement_fire_signals(portfolio_df, withdrawal_rate=0.04, market_drawdown_threshold=-0.20):
    signals = pd.DataFrame(index=portfolio_df.index)
    
    # Calculate daily returns
    portfolio_df['Daily_Return'] = portfolio_df['Portfolio_Value'].pct_change()
    
    # Calculate withdrawal amount based on the withdrawal rate
    withdrawal_amount = portfolio_df['Portfolio_Value'].shift(1) * withdrawal_rate
    
    # Determine the remaining portfolio value after withdrawal
    portfolio_df['Post_Withdrawal_Value'] = portfolio_df['Portfolio_Value'] - withdrawal_amount
    
    # Assess if market is in drawdown
    portfolio_df['Drawdown'] = (portfolio_df['Post_Withdrawal_Value'] / portfolio_df['Post_Withdrawal_Value'].rolling(window=252).max()) - 1
    portfolio_df['In_Drawdown'] = portfolio_df['Drawdown'] < market_drawdown_threshold
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[portfolio_df['In_Drawdown'], 'signal'] = 'short'
    signals.loc[(portfolio_df['Post_Withdrawal_Value'] > portfolio_df['Portfolio_Value'].shift(1)), 'signal'] = 'long'
    
    return signals
