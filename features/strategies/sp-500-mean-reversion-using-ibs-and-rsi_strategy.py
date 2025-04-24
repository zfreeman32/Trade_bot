
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

# S&P 500 Mean Reversion Strategy using IBS and RSI
def mean_reversion_ibsrsi_signals(stock_df, rsi_window=14, ibs_window=14, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi_indicator = RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi_indicator.rsi()

    # Calculate Internal Bar Strength (IBS)
    low_min = stock_df['Low'].rolling(window=ibs_window).min()
    high_max = stock_df['High'].rolling(window=ibs_window).max()
    signals['IBS'] = (stock_df['Close'] - low_min) / (high_max - low_min)

    # Initialize signal column
    signals['signal'] = 'neutral'

    # Generate trading signals based on RSI and IBS
    signals.loc[(signals['RSI'] < rsi_oversold) & (signals['IBS'] < 0.2), 'signal'] = 'long'
    signals.loc[(signals['RSI'] > rsi_overbought) & (signals['IBS'] > 0.8), 'signal'] = 'short'

    # Drop unnecessary columns
    signals.drop(['RSI', 'IBS'], axis=1, inplace=True)

    return signals
