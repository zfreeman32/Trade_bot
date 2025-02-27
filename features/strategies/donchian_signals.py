import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# Donchain Channel Signals
def donchian_signals(df, entry_length=40, exit_length=15, atr_length=20, atr_factor=2, atr_stop_factor=2, atr_average_type='simple'):
    signals = pd.DataFrame(index=df.index)

    # Ensure numeric conversion
    df['High'] = pd.to_numeric(df['High'], errors='coerce').ffill()
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce').ffill()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce').ffill()

    # Compute Donchian Channel Stops (ensuring proper alignment)
    signals['BuyStop'] = df['High'].rolling(window=entry_length, min_periods=1).max()
    signals['ShortStop'] = df['Low'].rolling(window=entry_length, min_periods=1).min()
    signals['CoverStop'] = df['High'].rolling(window=exit_length, min_periods=1).max()
    signals['SellStop'] = df['Low'].rolling(window=exit_length, min_periods=1).min()

    # Compute ATR
    atr = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_length).average_true_range()

    # Apply chosen smoothing type
    if atr_average_type == "simple":
        atr_smoothed = atr.rolling(window=atr_length, min_periods=1).mean()
    elif atr_average_type == "exponential":
        atr_smoothed = atr.ewm(span=atr_length, adjust=False).mean()
    elif atr_average_type == "wilder":
        atr_smoothed = atr.ewm(alpha=1/atr_length, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported ATR smoothing type '{atr_average_type}'")

    # Initialize signal columns
    signals['donchian_buy_signals'] = 0
    signals['donchian_sell_signal'] = 0

    # Apply ATR filter if necessary
    if atr_factor > 0:
        volatility_filter = atr_smoothed * atr_factor

        # Ensure all Series align properly
        buy_condition = (df['High'] > signals['BuyStop']) & ((df['High'].shift(1) - volatility_filter.shift(1)) > 0)
        sell_condition = (df['Low'] < signals['ShortStop']) & ((df['Low'].shift(1) - volatility_filter.shift(1)) < 0)

        signals.loc[buy_condition, 'donchian_buy_signals'] = 1
        signals.loc[sell_condition, 'donchian_sell_signal'] = 1
    else:
        signals.loc[df['High'] > signals['BuyStop'], 'donchian_buy_signals'] = 1
        signals.loc[df['Low'] < signals['ShortStop'], 'donchian_sell_signal'] = 1

    # Drop unnecessary columns
    signals.drop(columns=['CoverStop', 'SellStop', 'BuyStop', 'ShortStop'], inplace=True)

    return signals.fillna(0)
