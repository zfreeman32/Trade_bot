import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

#RSI Divergence strategy
def rsi_signals_with_divergence(stock_df, window=14, long_threshold=30, short_threshold=70, width=10):
    """
    Computes RSI signals with divergence detection.

    Returns:
    A DataFrame with 'RSI_signal' and 'RSI' values.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = rsi_indicator.rsi()

    # Generate RSI overbought/oversold signals
    signals['RSI_buy_signal'] = 0
    signals['RSI_sell_signal'] = 0
    signals.loc[signals['RSI'] < long_threshold, 'RSI_buy_signal'] = 1
    signals.loc[signals['RSI'] > short_threshold, 'RSI_sell_signal'] = 1

    return signals

# %%
# Mass Index
def mass_index_signals(stock_df, fast_window=9, slow_window=25):
    """
    Computes Mass Index reversal signals.

    Returns:
    A DataFrame with 'Mass_Index' and 'mass_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Mass Index
    mass_index = trend.MassIndex(stock_df['High'], stock_df['Low'], window_fast=fast_window, window_slow=slow_window)
    signals['Mass_Index'] = mass_index.mass_index()

    # Calculate Short & Long EMAs for trend direction
    short_ema = trend.EMAIndicator(stock_df['Close'], window=fast_window).ema_indicator()
    long_ema = trend.EMAIndicator(stock_df['Close'], window=slow_window).ema_indicator()

    # Set thresholds for Mass Index reversals
    reversal_bulge_threshold = 27
    reversal_bulge_exit_threshold = 26.5

    # Determine if the market is in a downtrend
    in_downtrend = short_ema < long_ema

    # Generate signals for reversals
    signals['mass_buy_signal'] = 0
    signals['mass_sell_signal'] = 0
    signals.loc[(in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_buy_signal'] = 1
    
    signals.loc[(~in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_sell_signal'] = 1

    return signals