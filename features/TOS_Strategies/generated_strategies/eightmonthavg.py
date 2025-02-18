import pandas as pd
import numpy as np

# Eight Month Average Strategy
def eight_month_avg_signals(stock_df, length=8):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate the simple moving average over the specified length
    signals['sma'] = stock_df['Close'].rolling(window=length).mean()
    # Initialize the signal column
    signals['eight_month_avg_signals'] = 'neutral'
    # Add a BUY_AUTO order where average crosses below the price
    signals.loc[(stock_df['Close'] > signals['sma']) & (stock_df['Close'].shift(1) <= signals['sma'].shift(1)), 'eight_month_avg_signals'] = 'buy'
    # Add a SELL_AUTO order where average crosses above the price
    signals.loc[(stock_df['Close'] < signals['sma']) & (stock_df['Close'].shift(1) >= signals['sma'].shift(1)), 'eight_month_avg_signals'] = 'sell'
    signals.drop(['sma'], axis=1, inplace=True)
    return signals

