
import pandas as pd
from ta import trend

# CTA Trend Following Strategy
def cta_trend_following_signals(price_df, short_window=50, long_window=200):
    """
    Implements a CTA trend-following strategy using moving averages.
    
    Parameters:
    price_df (pd.DataFrame): DataFrame containing 'Close' prices.
    short_window (int): The short moving average window (default is 50).
    long_window (int): The long moving average window (default is 200).
    
    Returns:
    pd.DataFrame: DataFrame with trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=price_df.index)
    signals['Short_MA'] = trend.sma_indicator(price_df['Close'], window=short_window)
    signals['Long_MA'] = trend.sma_indicator(price_df['Close'], window=long_window)
    
    signals['Signal'] = 'neutral'
    signals.loc[(signals['Short_MA'] > signals['Long_MA']), 'Signal'] = 'long'
    signals.loc[(signals['Short_MA'] < signals['Long_MA']), 'Signal'] = 'short'
    
    return signals[['Signal']]
