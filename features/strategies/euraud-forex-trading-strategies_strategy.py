
import pandas as pd
import numpy as np

# EURAUD Trading Strategy Based on Economic Indicators and Technical Signals
def euraud_trading_signals(data, short_window=14, long_window=50, rsi_period=14):
    """
    Generate trading signals for the EURAUD pair based on moving averages
    and RSI (Relative Strength Index) indicators.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing 'Close' prices for EURAUD
    short_window (int): Short moving average window
    long_window (int): Long moving average window
    rsi_period (int): Period for calculating RSI
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral')
    """
    
    signals = pd.DataFrame(index=data.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    signals['RSI'] = 100 - (100 / (1 + rs))
    
    # Generate trading signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['short_mavg'] > signals['long_mavg']) & (signals['RSI'] < 70), 'signal'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']) & (signals['RSI'] > 30), 'signal'] = 'short'
    
    return signals[['signal']]
