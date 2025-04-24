
import pandas as pd

# Larry Swedroe Portfolio Strategy
def swedroe_portfolio_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the percent change for stock and bond returns
    signals['Returns'] = stock_df['Close'].pct_change()
    
    # Calculate the moving averages for small-cap value stocks (e.g. VIOV, VBR)
    signals['small_cap_value'] = stock_df['SmallCapValue'].rolling(window=12).mean()
    signals['int_small_cap'] = stock_df['IntSmallCap'].rolling(window=12).mean()
    signals['emerging_markets'] = stock_df['EmergingMarkets'].rolling(window=12).mean()
    signals['bonds'] = stock_df['Bonds'].rolling(window=12).mean()

    # Generate signals based on the strategy
    signals['portfolio_signal'] = 'neutral'
    conditions_long = (
        (signals['small_cap_value'] > signals['small_cap_value'].shift(1)) & 
        (signals['int_small_cap'] > signals['int_small_cap'].shift(1)) & 
        (signals['emerging_markets'] > signals['emerging_markets'].shift(1))
    )
    
    conditions_short = (
        (signals['small_cap_value'] < signals['small_cap_value'].shift(1)) & 
        (signals['int_small_cap'] < signals['int_small_cap'].shift(1)) & 
        (signals['emerging_markets'] < signals['emerging_markets'].shift(1))
    )
    
    signals.loc[conditions_long, 'portfolio_signal'] = 'long'
    signals.loc[conditions_short, 'portfolio_signal'] = 'short'
    
    return signals[['portfolio_signal']]
