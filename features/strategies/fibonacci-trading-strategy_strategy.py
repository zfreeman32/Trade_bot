
import pandas as pd

# Fibonacci Retracement Trading Strategy
def fibonacci_retracement_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Fibonacci levels
    max_price = stock_df['Close'].rolling(window=window).max()
    min_price = stock_df['Close'].rolling(window=window).min()
    
    # Fibonacci levels
    levels = {
        'level_0': max_price,
        'level_1': max_price - (max_price - min_price) * 0.236,
        'level_2': max_price - (max_price - min_price) * 0.382,
        'level_3': max_price - (max_price - min_price) * 0.618,
        'level_4': min_price
    }
    
    # Initialize signals
    signals['fibonacci_signal'] = 'neutral'
    
    # Generate buy and sell signals based on Fibonacci retracement levels
    for level in levels:
        if stock_df['Close'].shift(1) < levels[level] and stock_df['Close'] >= levels[level]:
            signals.loc[signals.index[-1], 'fibonacci_signal'] = 'long'
        elif stock_df['Close'].shift(1) > levels[level] and stock_df['Close'] <= levels[level]:
            signals.loc[signals.index[-1], 'fibonacci_signal'] = 'short'

    return signals
