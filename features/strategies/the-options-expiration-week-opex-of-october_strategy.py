
import pandas as pd
import numpy as np

# Membership Plans Trading Strategy
def membership_plan_signals(stock_df, entry_threshold=1.05, exit_threshold=0.95):
    """
    Generates trading signals based on the membership plan trading strategy.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with 'Close' prices.
    entry_threshold (float): Entry threshold for long signals.
    exit_threshold (float): Exit threshold for short signals.

    Returns:
    DataFrame: A DataFrame containing signals ('long', 'short', 'neutral') based on thresholds.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate simple moving average (SMA) as a benchmark
    signals['SMA'] = stock_df['Close'].rolling(window=20).mean()
    
    # Determine when to buy (long) or sell (short) based on the thresholds
    signals['long_signal'] = np.where(stock_df['Close'] > signals['SMA'] * entry_threshold, 'long', 'neutral')
    signals['short_signal'] = np.where(stock_df['Close'] < signals['SMA'] * exit_threshold, 'short', 'neutral')

    # Combine signals into a single column
    signals['signal'] = 'neutral'
    signals.loc[signals['long_signal'] == 'long', 'signal'] = 'long'
    signals.loc[signals['short_signal'] == 'short', 'signal'] = 'short'

    # Clean up the data by only keeping necessary columns
    signals.drop(['long_signal', 'short_signal', 'SMA'], axis=1, inplace=True)

    return signals
