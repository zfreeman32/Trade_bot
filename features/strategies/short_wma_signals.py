import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# 13-26 MA STrategy
def short_wma_signals(stock_df, short_period=13, long_period=26):
    """
    Computes 13-26 Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()

    # Generate crossover signals
    signals['wma_buy_signal'] = 0
    signals['wma_sell_signal'] = 0
    signals.loc[
        (short_wma > long_wma) & (short_wma.shift(1) <= long_wma.shift(1)), 'wma_buy_signal'
    ] = 1
    
    signals.loc[
        (short_wma <= long_wma) & (short_wma.shift(1) > long_wma.shift(1)), 'wma_sell_signal'
    ] = 1

    return signals

# %%
# Donchain Channel
def donchian_channel_strategy(stock_df, window=20):
    """
    Computes Donchian Channel breakout signals.

    Returns:
    A DataFrame with 'dc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Donchian Channel
    donchian = volatility.DonchianChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Upper_Channel'] = donchian.donchian_channel_hband()
    signals['Lower_Channel'] = donchian.donchian_channel_lband()

    # Determine long and short signals (breakout strategy)
    signals['dc_buy_signal'] = 0
    signals['dc_sell_signal'] = 0
    signals.loc[stock_df['Close'] > signals['Upper_Channel'], 'dc_buy_signal'] = 1
    signals.loc[stock_df['Close'] < signals['Lower_Channel'], 'dc_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['Upper_Channel', 'Lower_Channel'], axis=1, inplace=True)

    return signals

#%% 
# Iron Bot
def ironbot_trend_filter(stock_df, z_length=40, analysis_window=44, high_trend_limit=23.6, 
                         low_trend_limit=78.6, use_ema=False, ema_length=200):
    """
    Implements the IronBot Statistical Trend Filter strategy.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close' prices.
    - z_length: Window for Z-score calculation.
    - analysis_window: Trend analysis period.
    - high_trend_limit, low_trend_limit: Fibonacci levels for trend confirmation.
    - use_ema: Whether to use an EMA filter.
    - ema_length: EMA period.

    Returns:
    - A DataFrame with trend levels and trading signals.
    """
    
    signals = pd.DataFrame(index=stock_df.index)

    # === Compute Z-Score ===
    stock_df['z_score'] = (stock_df['Close'] - stock_df['Close'].rolling(z_length).mean()) / stock_df['Close'].rolling(z_length).std()

    # === Compute Fibonacci Trend Levels ===
    stock_df['highest_high'] = stock_df['High'].rolling(analysis_window).max()
    stock_df['lowest_low'] = stock_df['Low'].rolling(analysis_window).min()
    stock_df['price_range'] = stock_df['highest_high'] - stock_df['lowest_low']

    stock_df['high_trend_level'] = stock_df['highest_high'] - stock_df['price_range'] * (high_trend_limit / 100)
    stock_df['trend_line'] = stock_df['highest_high'] - stock_df['price_range'] * 0.5
    stock_df['low_trend_level'] = stock_df['highest_high'] - stock_df['price_range'] * (low_trend_limit / 100)

    # === Compute EMA Filter ===
    if use_ema:
        ema = trend.EMAIndicator(stock_df['Close'], ema_length)
        stock_df['ema'] = ema.ema_indicator()
        stock_df['ema_bullish'] = stock_df['Close'] >= stock_df['ema']
        stock_df['ema_bearish'] = stock_df['Close'] <= stock_df['ema']
    else:
        stock_df['ema_bullish'] = True
        stock_df['ema_bearish'] = True

    # === Entry Conditions ===
    stock_df['can_long'] = (stock_df['Close'] >= stock_df['trend_line']) & (stock_df['Close'] >= stock_df['high_trend_level']) & (stock_df['z_score'] >= 0) & stock_df['ema_bullish']
    stock_df['can_short'] = (stock_df['Close'] <= stock_df['trend_line']) & (stock_df['Close'] <= stock_df['low_trend_level']) & (stock_df['z_score'] <= 0) & stock_df['ema_bearish']

    # === Generate Trading Signals ===
    signals['ironbot_buy_signal'] = 0
    signals['ironbot_sell_signal'] = 0
    signals.loc[stock_df['can_long'], 'ironbot_buy_signal'] = 1
    signals.loc[stock_df['can_short'], 'ironbot_sell_signal'] = 1

    return signals
