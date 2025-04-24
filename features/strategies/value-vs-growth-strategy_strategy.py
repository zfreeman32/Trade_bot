
import pandas as pd

# Value vs Growth Rotation Strategy
def value_growth_rotation_signals(stock_df, value_threshold=0.2, growth_threshold=0.15):
    """
    Generate trading signals for a Value vs Growth rotation strategy.
    
    Parameters:
    stock_df (pd.DataFrame): A DataFrame containing stock data with columns 'Close', 'PE_ratio', and 'Growth_rate'.
    value_threshold (float): The threshold for identifying 'value' stocks based on PE ratio.
    growth_threshold (float): The threshold for identifying 'growth' stocks based on growth rate.
    
    Returns:
    pd.DataFrame: A DataFrame containing the trading signals.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Determine if the stock is a value or growth stock
    signals['is_value'] = stock_df['PE_ratio'] < value_threshold
    signals['is_growth'] = stock_df['Growth_rate'] > growth_threshold
    
    # Initialize signal column
    signals['signal'] = 'neutral'
    
    # Generate buy/sell signals based on rotation criteria
    signals.loc[signals['is_value'], 'signal'] = 'long'  # Buy value stocks
    signals.loc[signals['is_growth'], 'signal'] = 'long'  # Buy growth stocks
    signals.loc[~signals['is_value'] & ~signals['is_growth'], 'signal'] = 'short'  # Sell if neither

    return signals[['signal']]
