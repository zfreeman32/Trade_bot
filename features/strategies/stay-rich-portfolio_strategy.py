
import pandas as pd
import numpy as np

# Stay Rich Portfolio Strategy
def stay_rich_portfolio_signals(asset_df, inflation_rate=0.04, max_drawdown=0.3):
    signals = pd.DataFrame(index=asset_df.index)
    
    # Calculate returns
    signals['returns'] = asset_df['Close'].pct_change()
    
    # Calculate rolling maximum and drawdown
    signals['rolling_max'] = signals['returns'].cummax()
    signals['drawdown'] = signals['rolling_max'] - signals['returns']
    
    # Define conditions for long, short, and neutral positions
    signals['signal'] = 'neutral'
    
    # Generate signals based on returns and drawdown conditions
    signals.loc[(signals['returns'] > inflation_rate) & (signals['drawdown'] <= max_drawdown), 'signal'] = 'long'
    signals.loc[(signals['returns'] < inflation_rate) & (signals['drawdown'] > max_drawdown), 'signal'] = 'short'
    
    return signals[['signal']]

