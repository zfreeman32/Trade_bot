
import pandas as pd

# 3-Day Weekend Trading Strategy
def three_day_weekend_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Create a column to identify if the next day is the last trading day before a 3-day weekend
    signals['is_weekend'] = stock_df.index.to_series().dt.dayofweek.isin([4])  # Friday is day 4
    
    # Create a column for the close price shifted by -1 (next day)
    signals['next_close'] = stock_df['Close'].shift(-1)
    
    # Generate signals based on the trading rules of rising prices before a 3-day weekend
    signals['signal'] = 'neutral'
    signals.loc[(signals['is_weekend']) & (stock_df['Close'] < signals['next_close']), 'signal'] = 'long'
    signals.loc[(signals['is_weekend']) & (stock_df['Close'] >= signals['next_close']), 'signal'] = 'short'
    
    # Clean up the DataFrame
    signals.drop(['is_weekend', 'next_close'], axis=1, inplace=True)
    
    return signals
