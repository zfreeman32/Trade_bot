
import pandas as pd

# Russell 2000 Rebalancing Strategy
def russell_2000_rebalancing_signals(stock_df):
    # Ensure the index is a datetime type
    stock_df.index = pd.to_datetime(stock_df.index)
    
    # Create a signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    
    # Retrieve the year and month for each date
    signals['year'] = stock_df.index.year
    signals['month'] = stock_df.index.month
    
    # Initialize a column for signals
    signals['rebalance_signal'] = 'neutral'
    
    # Define rebalancing period: Buy on the close of the first trading day after June 23rd
    buy_date = pd.to_datetime(stock_df[(signals['month'] == 6) & (stock_df.index.day > 23)].index.min())
    
    # Define sell date: Sell on the close of the first trading day of July
    sell_date = pd.to_datetime(stock_df[(signals['month'] == 7)].index.min())
    
    # Looking for the buy signal
    if buy_date in stock_df.index:
        signals.loc[buy_date, 'rebalance_signal'] = 'long'
    else:
        signals.loc[(stock_df.index > buy_date) & (stock_df.index < sell_date), 'rebalance_signal'] = 'long'
    
    # Looking for the sell signal
    if sell_date in stock_df.index:
        signals.loc[sell_date, 'rebalance_signal'] = 'short'
    
    # Ensure that we have a 'neutral' signal outside of trade periods
    signals['rebalance_signal'] = signals['rebalance_signal'].where(signals['rebalance_signal'] != 'neutral', 'neutral')
    
    return signals[['rebalance_signal']]
