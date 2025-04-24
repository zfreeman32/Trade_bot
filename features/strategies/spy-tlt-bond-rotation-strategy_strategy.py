
import pandas as pd

# Monthly Momentum Bond Rotation Strategy
def bond_rotation_signals(price_data):
    """
    Generates trading signals based on monthly momentum for SPY and TLT.
    
    Parameters:
    price_data (pd.DataFrame): DataFrame containing 'SPY' and 'TLT' price data with Date as index.
    
    Returns:
    pd.DataFrame: DataFrame containing the trading signals ('long', 'short', 'neutral').
    """
    
    # Calculate monthly returns
    monthly_returns = price_data.resample('M').ffill().pct_change()
    
    # Calculate the previous month's return for SPY and TLT
    previous_month_returns = monthly_returns.shift(1)
    
    signals = pd.DataFrame(index=monthly_returns.index)
    
    # Initialize signals based on previous month returns
    signals['signal'] = 'neutral'
    
    # Generate signals
    signals.loc[previous_month_returns['SPY'] > previous_month_returns['TLT'], 'signal'] = 'long'
    signals.loc[previous_month_returns['SPY'] <= previous_month_returns['TLT'], 'signal'] = 'short'

    return signals[['signal']]
