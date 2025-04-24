
import pandas as pd
import numpy as np

# EUR/USD Trading Strategy
def eurusd_trading_signals(price_data, fib_levels, rsi_window=14, z_score_threshold=1):
    """
    Generate trading signals for EUR/USD based on Fibonacci retracement levels and 
    RSI (Relative Strength Index) analysis.
    
    Parameters:
    - price_data: DataFrame containing 'Close' prices of EUR/USD.
    - fib_levels: Tuple containing the start and end price values for Fibonacci calculation.
    - rsi_window: The lookback period for the RSI.
    - z_score_threshold: The threshold for Z-score to determine entry conditions.
    
    Returns:
    - DataFrame with trading signals ('long', 'short', 'neutral').
    """
    
    signals = pd.DataFrame(index=price_data.index)
    
    # Calculate Fibonacci retracement levels
    start_price, end_price = fib_levels
    fib_retracement = {
        "level_0": end_price,
        "level_1": end_price - (start_price - end_price) * 0.236,
        "level_2": end_price - (start_price - end_price) * 0.382,
        "level_3": end_price - (start_price - end_price) * 0.618,
        "level_4": start_price
    }
    
    # Calculate RSI
    delta = price_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    rs = gain / loss
    price_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Z-score
    price_mean = price_data['Close'].rolling(window=30).mean()
    price_std = price_data['Close'].rolling(window=30).std()
    price_data['Z-Score'] = (price_data['Close'] - price_mean) / price_std
    
    # Generate trading signals
    signals['signal'] = 'neutral'
    signals.loc[(price_data['Close'] <= fib_retracement['level_3']) & (price_data['RSI'] < 30) & (price_data['Z-Score'] < -z_score_threshold), 'signal'] = 'long'
    signals.loc[(price_data['Close'] >= fib_retracement['level_1']) & (price_data['RSI'] > 70) & (price_data['Z-Score'] > z_score_threshold), 'signal'] = 'short'
    
    return signals
