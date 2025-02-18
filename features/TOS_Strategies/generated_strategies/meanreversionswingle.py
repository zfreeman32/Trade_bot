import pandas as pd

# Function to detect uptrend based on moving averages
def detect_uptrend(close_prices, min_length, min_range_for_uptrend):
    rolling_min = close_prices.rolling(window=min_length).min()
    rolling_max = close_prices.rolling(window=min_length).max()
    return (rolling_max - rolling_min) > min_range_for_uptrend

# Function to detect pullbacks (retracements from highs)
def detect_pullback(close_prices, uptrend_signal, tolerance):
    rolling_high = close_prices.rolling(window=10).max()
    return (rolling_high - close_prices) > tolerance

# Function to detect upward movement after pullback
def detect_up_move(close_prices, pullback_signal, min_up_move):
    price_diff = close_prices.diff()
    return (price_diff > min_up_move) & pullback_signal

# Function to limit the pattern length
def limit_pattern_length(signals, max_length):
    signals['pattern_length'] = signals['mean_reversion_swing_le'].groupby((signals['mean_reversion_swing_le'] != signals['mean_reversion_swing_le'].shift()).cumsum()).cumcount() + 1
    signals.loc[signals['pattern_length'] > max_length, 'mean_reversion_swing_le'] = 'neutral'
    return signals.drop(columns=['pattern_length'])

# Mean Reversion Swing Strategy (Long Entries)
def mean_reversion_swing_le(stock_df, min_length=20, max_length=400, min_range_for_uptrend=5.0,
                            min_up_move=0.5, tolerance=1.0):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Compute trend-based conditions
    signals['uptrend'] = detect_uptrend(stock_df['Close'], min_length, min_range_for_uptrend)
    signals['pullback'] = detect_pullback(stock_df['Close'], signals['uptrend'], tolerance)
    signals['up_move'] = detect_up_move(stock_df['Close'], signals['pullback'], min_up_move)
    
    # Generate trade signals
    signals['mean_reversion_swing_le'] = 'neutral'
    signals.loc[signals['uptrend'] & signals['pullback'] & signals['up_move'], 'mean_reversion_swing_le'] = 'long'
    
    # Ensure the pattern length does not exceed max_length
    signals = limit_pattern_length(signals, max_length)
    
    # Drop intermediate calculation columns
    signals = signals.drop(['uptrend', 'pullback', 'up_move'], axis=1)

    return signals

