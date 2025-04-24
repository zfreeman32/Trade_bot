
import pandas as pd

# Silver Seasonal Strategy Signals
def silver_seasonal_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['month'] = stock_df.index.month
    signals['seasonal_signal'] = 'neutral'
    
    # Best months for long positions
    long_months = [1, 7]  # January and July
    
    # Assign signals based on the month
    signals.loc[signals['month'].isin(long_months), 'seasonal_signal'] = 'long'
    
    # Months to avoid holding
    short_months = [6, 9]  # June and September

    signals.loc[signals['month'].isin(short_months), 'seasonal_signal'] = 'short'
    
    return signals[['seasonal_signal']]
