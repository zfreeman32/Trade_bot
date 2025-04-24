
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

# Seasonal Strategy Based on Historical Data-Driven Indicators
def seasonal_strategy_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate the average return over the specified window
    signals['Avg_Return'] = stock_df['Close'].pct_change(periods=window).rolling(window).mean()

    # Create signals based on the average return
    signals['seasonal_signal'] = 'neutral'
    signals.loc[signals['Avg_Return'] > 0, 'seasonal_signal'] = 'long'
    signals.loc[signals['Avg_Return'] < 0, 'seasonal_signal'] = 'short'
    
    # Drop the Avg_Return column if not needed for final output
    signals.drop(['Avg_Return'], axis=1, inplace=True)
    
    return signals
