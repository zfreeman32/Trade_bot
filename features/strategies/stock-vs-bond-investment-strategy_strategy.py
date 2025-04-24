
import pandas as pd

# Stocks vs. Bonds Investment Strategy
def stocks_vs_bonds_signals(stock_df, bond_df, stock_threshold=0.05, bond_threshold=0.03):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate returns
    stock_returns = stock_df['Close'].pct_change()
    bond_returns = bond_df['Close'].pct_change()
    
    # Generate signals based on thresholds
    signals['signal'] = 'neutral'
    signals.loc[stock_returns > stock_threshold, 'signal'] = 'long'
    signals.loc[bond_returns > bond_threshold, 'signal'] = 'short'
    signals.loc[(stock_returns <= stock_threshold) & (bond_returns <= bond_threshold), 'signal'] = 'neutral'
    
    # Clean up signals DataFrame
    signals.drop(['signal'], axis=1, inplace=True)
    
    return signals
