
import pandas as pd

# David Swensen Portfolio Strategy
def swensen_portfolio_signals(asset_df):
    """
    Generates signals based on the David Swensen Portfolio allocation strategy.
    
    The allocation is as follows:
    - 30% Total Stock Market
    - 15% International Stock Market
    - 5% Emerging Markets
    - 15% Intermediate Treasury Bonds
    - 15% Treasury Inflation-Protected Securities (TIPS)
    - 20% Real Estate Investment Trusts (REITs)

    Parameters:
    asset_df (pd.DataFrame): Dataframe containing asset prices with columns
                             ['Total Stock Market', 'International Stocks', 
                              'Emerging Markets', 'Intermediate Bonds',
                              'TIPS', 'REITs']
    
    Returns:
    pd.DataFrame: A DataFrame with a 'portfolio_signal' column indicating
                  the allocation strategy.
    """
    allocations = {
        'Total Stock Market': 0.30,
        'International Stocks': 0.15,
        'Emerging Markets': 0.05,
        'Intermediate Bonds': 0.15,
        'TIPS': 0.15,
        'REITs': 0.20
    }
    
    portfolio_values = pd.Series(index=asset_df.index)
    
    for asset, allocation in allocations.items():
        portfolio_values += asset_df[asset] * allocation

    # Create signals based on portfolio performance
    signals = pd.DataFrame(index=asset_df.index)
    signals['portfolio_return'] = portfolio_values.pct_change()
    signals['portfolio_signal'] = 'neutral'
    
    signals.loc[signals['portfolio_return'] > 0, 'portfolio_signal'] = 'long'
    signals.loc[signals['portfolio_return'] < 0, 'portfolio_signal'] = 'short'
    
    return signals[['portfolio_signal']]

