
import pandas as pd

# Best Performing Markets Strategy
def best_performing_markets_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['performance_signal'] = 'neutral'
    
    # Define the top performing markets as per the description
    top_performing_markets = ['Australia', 'United States', 'South Africa', 'New Zealand', 'Denmark']
    
    # Assuming stock_df contains a column named 'Country' for the stock's country
    # and a column 'Annual_Return' for the annual return
    for market in top_performing_markets:
        signals.loc[(stock_df['Country'] == market) & (stock_df['Annual_Return'] > 6.5), 'performance_signal'] = 'long'
        signals.loc[(stock_df['Country'] == market) & (stock_df['Annual_Return'] <= 6.5), 'performance_signal'] = 'short'
    
    return signals
