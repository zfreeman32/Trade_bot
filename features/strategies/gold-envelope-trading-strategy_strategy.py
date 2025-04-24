
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility

# Quantified Strategies Membership Trading Signals
def membership_strategy_signals(stock_df, window=14):
    """
    This function generates trading signals based on quantified patterns
    and anomalies indicating abnormal returns. The strategy uses
    momentum indicators and establishes long and short positions based
    on clear quantifiable entry and exit rules.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    window (int): Window for calculating momentum indicator.

    Returns:
    pd.DataFrame: DataFrame with columns for signals: 'membership_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate momentum indicator (e.g., RSI)
    momentum_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['momentum'] = momentum_indicator.rsi()

    # Initialize signals as 'neutral'
    signals['membership_signal'] = 'neutral'

    # Long signal: when momentum crosses above threshold (e.g., 70)
    signals.loc[(signals['momentum'] > 70) & (signals['momentum'].shift(1) <= 70), 'membership_signal'] = 'long'

    # Short signal: when momentum crosses below threshold (e.g., 30)
    signals.loc[(signals['momentum'] < 30) & (signals['momentum'].shift(1) >= 30), 'membership_signal'] = 'short'

    return signals[['membership_signal']]
