
import pandas as pd

# Dogs of the Dow Trading Strategy
def dogs_of_the_dow_signals(stock_df, dividend_yields):
    """
    Generate trading signals based on the Dogs of the Dow strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock data with a 'Close' price.
    dividend_yields (dict): A dictionary with stock tickers as keys and their dividend yields as values.
    
    Returns:
    pd.DataFrame: DataFrame containing the trading signals ('long', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Sort tickers by dividend yield and select the top 10
    top_dividend_stocks = sorted(dividend_yields, key=dividend_yields.get, reverse=True)[:10]
    
    # Create a list to hold signals
    signals['Signal'] = 'neutral'
    
    # Check if the stock is in the top 10 and generate signals
    for ticker in top_dividend_stocks:
        if ticker in stock_df.columns:
            signals.loc[stock_df[ticker].notnull(), 'Signal'] = 'long'
    
    return signals
