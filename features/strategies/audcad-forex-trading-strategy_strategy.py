
import pandas as pd
import numpy as np
from ta import momentum, trend

# AUDCAD Forex Trading Strategy
def audcad_signals(forex_df, window=14, threshold=0.001):
    """
    Generates trading signals for the AUDCAD forex pair using a momentum strategy.
    Buy signal occurs when the price increases significantly above the previous close,
    and a sell signal occurs when it decreases significantly below the previous close.

    Parameters:
    forex_df (DataFrame): DataFrame containing the forex prices with 'Close' column.
    window (int): Lookback period for momentum calculation.
    threshold (float): Threshold for generating buy/sell signals.

    Returns:
    DataFrame: A DataFrame containing signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=forex_df.index)
    signals['Close'] = forex_df['Close']
    
    # Calculate momentum
    signals['Momentum'] = signals['Close'].diff(window)
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[signals['Momentum'] > threshold, 'signal'] = 'long'
    signals.loc[signals['Momentum'] < -threshold, 'signal'] = 'short'
    
    return signals[['signal']]
