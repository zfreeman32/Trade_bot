
import pandas as pd

# Seasonal Trading Strategy for DAX 40 and German Stocks
def seasonal_dax_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Extract month from the index
    signals['month'] = stock_df.index.month
    
    # Define trading signals based on the seasonal strategy
    signals['seasonal_signal'] = 'neutral'
    
    # Set long signal for October, November, December (4th quarter)
    signals.loc[signals['month'].isin([10, 11, 12]), 'seasonal_signal'] = 'long'
    
    # Set short signal for January to September (1st to 3rd quarters)
    signals.loc[signals['month'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), 'seasonal_signal'] = 'short'
    
    # Clean up the DataFrame, removing the month column
    signals.drop(['month'], axis=1, inplace=True)
    
    return signals
