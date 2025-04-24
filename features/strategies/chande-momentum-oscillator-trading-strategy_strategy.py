
import pandas as pd
import numpy as np

# Chande Momentum Oscillator Strategy
def cmo_signals(stock_df, window=14):
    """
    Generates trading signals based on the Chande Momentum Oscillator.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    window (int): The lookback period for calculating the CMO.

    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Chande Momentum Oscillator
    delta = stock_df['Close'].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    
    cmo = ((avg_gain - avg_loss) / (avg_gain + avg_loss)) * 100

    signals['CMO'] = cmo
    signals['cmo_signal'] = 'neutral'
    
    # Define trading signals based on CMO levels
    signals.loc[(signals['CMO'] > 50), 'cmo_signal'] = 'long'
    signals.loc[(signals['CMO'] < -50), 'cmo_signal'] = 'short'

    return signals[['cmo_signal']]
