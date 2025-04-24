
import pandas as pd
import numpy as np

# Copper vs Stocks Trading Strategy
def copper_vs_stocks_signals(copper_df, stocks_df):
    signals = pd.DataFrame(index=copper_df.index)
    
    # Calculate the daily returns for copper and stocks
    copper_returns = copper_df['Close'].pct_change()
    stocks_returns = stocks_df['Close'].pct_change()

    # Create a moving average for copper prices
    copper_mavg = copper_df['Close'].rolling(window=5).mean()

    # Determine signals based on copper price movement and stock market response
    signals['copper_signal'] = 'neutral'
    signals.loc[(copper_df['Close'] > copper_mavg) & (copper_returns > 0) & (stocks_returns > 0), 'copper_signal'] = 'long'
    signals.loc[(copper_df['Close'] < copper_mavg) & (copper_returns < 0) & (stocks_returns < 0), 'copper_signal'] = 'short'
    
    return signals[['copper_signal']]
