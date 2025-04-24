
import pandas as pd
import numpy as np

# Relative Momentum Index (RMI) Strategy
def rmi_signals(stock_df, n=14, threshold_overbought=70, threshold_oversold=30):
    """
    Calculate the Relative Momentum Index (RMI) trading signals.

    Parameters:
    - stock_df: DataFrame containing stock data with a 'Close' column.
    - n: The lookback period for RMI calculation.
    - threshold_overbought: The threshold above which the market is considered overbought.
    - threshold_oversold: The threshold below which the market is considered oversold.

    Returns:
    - DataFrame with signals indicating 'long', 'short', or 'neutral'.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the RMI
    delta = stock_df['Close'].diff(n)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=n).mean()
    avg_loss = pd.Series(loss).rolling(window=n).mean()
    
    rs = avg_gain / avg_loss
    rmi = 100 - (100 / (1 + rs))
    
    signals['RMI'] = rmi
    signals['rmi_signal'] = 'neutral'
    signals.loc[(signals['RMI'] > threshold_overbought), 'rmi_signal'] = 'short'
    signals.loc[(signals['RMI'] < threshold_oversold), 'rmi_signal'] = 'long'
    
    signals.drop(['RMI'], axis=1, inplace=True)
    return signals
