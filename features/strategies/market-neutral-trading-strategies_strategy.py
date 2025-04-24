
import pandas as pd
import numpy as np

# Market-Neutral Strategy
def market_neutral_signals(stock_df1, stock_df2):
    """
    Generate market-neutral trading signals based on long positions in one stock
    and short positions in another stock.

    Parameters:
    stock_df1 (pd.DataFrame): DataFrame containing historical prices for stock 1.
    stock_df2 (pd.DataFrame): DataFrame containing historical prices for stock 2.

    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df1.index)
    
    # Calculate returns for both stocks
    stock_df1['Returns'] = stock_df1['Close'].pct_change()
    stock_df2['Returns'] = stock_df2['Close'].pct_change()
    
    # Calculate the cumulative returns for ranking
    signals['Cumulative_Returns_Stock1'] = (1 + stock_df1['Returns']).cumprod()
    signals['Cumulative_Returns_Stock2'] = (1 + stock_df2['Returns']).cumprod()
    
    # Determine signals based on returns.
    # Long stock_df1 and short stock_df2 if stock_df1 has better returns
    signals['signal'] = 'neutral'
    signals.loc[signals['Cumulative_Returns_Stock1'] > signals['Cumulative_Returns_Stock2'], 'signal'] = 'long'
    signals.loc[signals['Cumulative_Returns_Stock1'] < signals['Cumulative_Returns_Stock2'], 'signal'] = 'short'
    
    return signals[['signal']]
