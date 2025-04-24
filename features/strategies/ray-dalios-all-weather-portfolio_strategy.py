
import pandas as pd

# Ray Dalio's All Weather Portfolio Strategy
def all_weather_portfolio_signals(stock_df, bond_df, commodity_df, rebalance_frequency='M'):
    """
    Generate trading signals based on Ray Dalio's All Weather Portfolio strategy.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock prices with a 'Close' column.
    bond_df (pd.DataFrame): DataFrame containing bond prices with a 'Close' column.
    commodity_df (pd.DataFrame): DataFrame containing commodity prices with a 'Close' column.
    rebalance_frequency (str): Frequency of rebalancing. Default is 'M' for monthly.

    Returns:
    pd.DataFrame: DataFrame containing signals for stocks, bonds, and commodities.
    """
    # Calculate the long-term portfolio weights
    weights = {'Stocks': 0.30, 'Bonds': 0.55, 'Commodities': 0.15}

    # Calculate returns
    stock_returns = stock_df['Close'].pct_change()
    bond_returns = bond_df['Close'].pct_change()
    commodity_returns = commodity_df['Close'].pct_change()

    # Create a DataFrame to hold the signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Market'] = 'neutral'

    # Daily portfolio returns based on weights
    combined_returns = (weights['Stocks'] * stock_returns + 
                        weights['Bonds'] * bond_returns + 
                        weights['Commodities'] * commodity_returns)

    # Generate signals based on moving average of portfolio returns
    signals['Signal'] = combined_returns.rolling(window=30).mean()
    signals.loc[signals['Signal'] > 0, 'Market'] = 'long'
    signals.loc[signals['Signal'] < 0, 'Market'] = 'short'

    # Resample signals based on rebalance frequency
    if rebalance_frequency == 'M':
        signals = signals.resample('M').last()

    return signals[['Market']]
