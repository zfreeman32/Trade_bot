
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Seasonal Trading Strategy
def seasonal_trading_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the average return of the past 'window' periods
    signals['Returns'] = stock_df['Close'].pct_change()
    signals['Avg_Return'] = signals['Returns'].rolling(window=window).mean()

    # Create trading signals based on the average return
    signals['signal'] = 'neutral'
    signals.loc[signals['Avg_Return'] > 0, 'signal'] = 'long'
    signals.loc[signals['Avg_Return'] < 0, 'signal'] = 'short'
    
    return signals[['signal']]
