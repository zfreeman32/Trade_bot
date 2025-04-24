
import pandas as pd

# 60/40 Portfolio Strategy
def sixty_forty_portfolio_signals(equities_df, bonds_df, allocation_eq=0.6, allocation_bonds=0.4):
    signals = pd.DataFrame(index=equities_df.index)
    # Calculate the returns for equities and bonds
    signals['Equities_Returns'] = equities_df['Close'].pct_change()
    signals['Bonds_Returns'] = bonds_df['Close'].pct_change()
    
    # Calculate the portfolio returns based on the allocated weights
    signals['Portfolio_Returns'] = (signals['Equities_Returns'] * allocation_eq) + (signals['Bonds_Returns'] * allocation_bonds)

    # Generate signals based on the portfolio returns
    signals['signal'] = 'neutral'
    signals.loc[signals['Portfolio_Returns'] > 0, 'signal'] = 'long'
    signals.loc[signals['Portfolio_Returns'] < 0, 'signal'] = 'short'
    
    # Forward fill the signals to persist the last known status
    signals['signal'] = signals['signal'].ffill().fillna('neutral')
    
    return signals[['signal']]
