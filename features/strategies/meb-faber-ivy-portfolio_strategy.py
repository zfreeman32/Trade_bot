
import pandas as pd
import numpy as np

# Meb Faber Ivy Portfolio Strategy
def ivy_portfolio_signals(stock_df, stock_weight=0.6, bond_weight=0.2, commodity_weight=0.2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Assuming stock_df contains columns for stocks, bonds, and commodities
    total_investment = stock_weight + bond_weight + commodity_weight
    
    # Generate normalized weights
    signals['stock_weight'] = stock_weight / total_investment
    signals['bond_weight'] = bond_weight / total_investment
    signals['commodity_weight'] = commodity_weight / total_investment
    
    # Create trading signals based on hypothetical allocation strategy
    signals['portfolio_signal'] = 'neutral'
    
    # For simplicity, let's assume we consider a "long" position when stocks are trending up
    signals.loc[(stock_df['Stocks'].pct_change() > 0), 'portfolio_signal'] = 'long'
    
    # Conversely, we might want to "short" or exit when stocks are trending down
    signals.loc[(stock_df['Stocks'].pct_change() < 0), 'portfolio_signal'] = 'short'
    
    return signals[['stock_weight', 'bond_weight', 'commodity_weight', 'portfolio_signal']]
