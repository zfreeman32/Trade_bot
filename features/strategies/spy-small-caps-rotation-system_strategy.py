
import pandas as pd

# SPY Small Caps Rotation System Strategy
def spy_small_caps_rotation(stock_df, market_condition_threshold=100):
    """
    Generates trading signals for SPY Small Caps Rotation System.
    
    Parameters:
    - stock_df: DataFrame containing stock data with 'Close' prices
    - market_condition_threshold: threshold for market conditions to determine when to rotate into small caps
    
    Returns:
    - DataFrame with trading signals ('long', 'short', 'neutral')
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate market condition as an example (this could be any relevant indicator)
    signals['Market_Condition'] = stock_df['Close'].rolling(window=30).mean()  # Simple Moving Average as market condition

    # Initialize signals
    signals['signal'] = 'neutral'
    
    # Generate signals based on market conditions
    signals.loc[signals['Market_Condition'] > market_condition_threshold, 'signal'] = 'long'
    signals.loc[signals['Market_Condition'] <= market_condition_threshold, 'signal'] = 'short'
    
    return signals[['signal']]
