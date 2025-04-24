
import pandas as pd
import numpy as np

# Pretty Good Oscillator (PGO) Strategy
def pgo_signals(stock_df, window=14, atr_window=14):
    """
    Generate trading signals based on the Pretty Good Oscillator (PGO) strategy.

    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with 'Close' and 'High', 'Low' columns.
    window (int): The period for the moving average used in PGO calculation.
    atr_window (int): The period for the Average True Range (ATR) calculation.

    Returns:
    DataFrame: DataFrame with signals: 'long', 'short', 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate the moving average
    signals['Moving_Average'] = stock_df['Close'].rolling(window=window).mean()

    # Calculate the Average True Range (ATR)
    high_low = stock_df['High'] - stock_df['Low']
    high_close_prev = abs(stock_df['High'] - stock_df['Close'].shift(1))
    low_close_prev = abs(stock_df['Low'] - stock_df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    signals['ATR'] = ranges.max(axis=1).rolling(window=atr_window).mean()

    # Calculate the PGO
    signals['PGO'] = (stock_df['Close'] - signals['Moving_Average']) / signals['ATR']

    # Generate signals based on PGO
    signals['pgo_signal'] = 'neutral'
    signals.loc[(signals['PGO'] > 0) & (signals['PGO'].shift(1) <= 0), 'pgo_signal'] = 'long'
    signals.loc[(signals['PGO'] < 0) & (signals['PGO'].shift(1) >= 0), 'pgo_signal'] = 'short'

    return signals[['pgo_signal']]
