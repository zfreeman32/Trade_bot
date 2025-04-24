
import pandas as pd

# Asset Allocation Strategy
def asset_allocation_signals(stock_df, stock_weight=0.6, bond_weight=0.2, cash_weight=0.2):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Asset_Class'] = 'neutral'  # Default to neutral

    # Calculate the returns for stocks, bonds, and cash
    stock_returns = stock_df['Close'].pct_change().fillna(0)
    
    # Simulate bonds returns (for simplicity, merely using a fixed or expected return)
    bond_returns = pd.Series(0.02, index=stock_df.index)  # Example fixed return for bonds
    
    # Simulating cash returns
    cash_returns = pd.Series(0.01, index=stock_df.index)  # Example fixed return for cash

    # Calculate combined weighted returns
    combined_returns = (stock_weight * stock_returns) + (bond_weight * bond_returns) + (cash_weight * cash_returns)

    # Define the signals based on combined returns
    signals.loc[combined_returns > 0, 'Asset_Class'] = 'long'
    signals.loc[combined_returns < 0, 'Asset_Class'] = 'short'

    return signals
