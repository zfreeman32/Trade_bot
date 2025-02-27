import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# High Volume Points
def high_volume_points(stock_df, left_bars=15, filter_vol=2.0, lookback_period=300):
    """
    Identifies high-volume pivot points in price data.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close', 'Volume'.
    - left_bars: Number of bars before & after a pivot high/low for validation.
    - filter_vol: Minimum volume threshold to filter significant pivot points.
    - lookback_period: Number of bars to consider for percentile volume ranking.

    Returns:
    - A DataFrame with high-volume pivot points and liquidity grab signals.
    """

    signals = pd.DataFrame(index=stock_df.index)

    # === Compute Pivot Highs & Lows ===
    stock_df['pivot_high'] = stock_df['High'][(stock_df['High'] == stock_df['High'].rolling(window=left_bars*2+1, center=True).max())]
    stock_df['pivot_low'] = stock_df['Low'][(stock_df['Low'] == stock_df['Low'].rolling(window=left_bars*2+1, center=True).min())]

    # === Compute Normalized Volume ===
    stock_df['rolling_volume'] = stock_df['Volume'].rolling(window=left_bars * 2).sum()
    reference_vol = stock_df['rolling_volume'].rolling(lookback_period).quantile(0.95)
    stock_df['norm_vol'] = stock_df['rolling_volume'] / reference_vol * 5  # Normalize between 0-6 scale

    # === Apply Volume Filter ===
    stock_df['valid_pivot_high'] = (stock_df['pivot_high'].notna()) & (stock_df['norm_vol'] > filter_vol)
    stock_df['valid_pivot_low'] = (stock_df['pivot_low'].notna()) & (stock_df['norm_vol'] > filter_vol)

    # === Detect Liquidity Grabs ===
    stock_df['liquidity_grab_high'] = stock_df['valid_pivot_high'] & (stock_df['Close'] < stock_df['pivot_high'].shift(1))
    stock_df['liquidity_grab_low'] = stock_df['valid_pivot_low'] & (stock_df['Close'] > stock_df['pivot_low'].shift(1))

    # === Store Relevant Features for ML Training ===
    signals['high_volume_pivot'] = stock_df['valid_pivot_high'] | stock_df['valid_pivot_low']
    signals['liquidity_grab'] = stock_df['liquidity_grab_high'] | stock_df['liquidity_grab_low']

    return signals
