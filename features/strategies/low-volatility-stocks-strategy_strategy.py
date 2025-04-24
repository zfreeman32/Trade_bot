
import pandas as pd

# Low Volatility Stocks Strategy
def low_volatility_signals(stock_df, beta_threshold=1.0):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the beta of the stock
    # Assuming 'market_returns' is a DataFrame with market returns
    # Using a placeholder for market returns; in practice, you would get this data
    market_returns = stock_df['Market']  # This should be your market index returns
    stock_returns = stock_df['Close'].pct_change()  # Calculate daily returns of the stock
    cov_matrix = stock_returns.rolling(window=60).cov(market_returns.rolling(window=60))  # Rolling covariance
    market_var = market_returns.rolling(window=60).var()  # Rolling variance of the market
    stock_beta = cov_matrix / market_var  # Calculate beta

    # Signal generation based on beta
    signals['beta'] = stock_beta
    signals['signal'] = 'neutral'
    signals.loc[signals['beta'] < beta_threshold, 'signal'] = 'long'
    signals.loc[signals['beta'] >= beta_threshold, 'signal'] = 'short'
    
    return signals[['signal']]
