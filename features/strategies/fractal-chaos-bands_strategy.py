
import pandas as pd

# Fractal Chaos Bands Strategy
def fractal_chaos_bands_signals(stock_df, window=5):
    """
    This function implements the Fractal Chaos Bands trading strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame with 'High', 'Low', and 'Close' prices
    window (int): Number of periods to use for calculating fractals

    Returns:
    pd.DataFrame: DataFrame with trading signals
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Fractal Highs and Lows
    stock_df['Fractal_High'] = stock_df['High'].rolling(window=window, center=True).max()
    stock_df['Fractal_Low'] = stock_df['Low'].rolling(window=window, center=True).min()
    
    # Create signals based on fractals
    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > stock_df['Fractal_High'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] < stock_df['Fractal_Low'], 'signal'] = 'short'
    
    return signals[['signal']]
