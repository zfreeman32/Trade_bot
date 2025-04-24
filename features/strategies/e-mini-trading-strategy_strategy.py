
import pandas as pd
import numpy as np

# E-mini Trading Strategy
def e_mini_signals(data, short_window=3, long_window=10):
    """
    Generate trading signals for E-mini contracts based on a simple moving average crossover strategy.
    
    Parameters:
    - data: DataFrame containing historical price data (with a 'Close' column).
    - short_window: The short moving average window (default=3).
    - long_window: The long moving average window (default=10).
    
    Returns:
    - DataFrame with trading signals: 'long', 'short', 'neutral'.
    """
    signals = pd.DataFrame(index=data.index)
    
    # Calculate short and long moving averages
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Initialize the signals column
    signals['signal'] = 0
    
    # Generate long and short signals
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Create a 'position' column to determine entry/exit points
    signals['position'] = signals['signal'].diff()
    
    # Assign trading signals based on position
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trade_signal'] = 'long'  # Buy signal
    signals.loc[signals['position'] == -1, 'trade_signal'] = 'short' # Sell signal
    
    return signals[['trade_signal']]
