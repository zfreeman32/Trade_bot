#%%
import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange

def donchian_signals(df, entry_length=40, exit_length=15, atr_length=20, atr_factor=2, atr_stop_factor=2, atr_average_type='simple'):
    signals = pd.DataFrame(index=df.index)
    
    # Compute Donchian Channel Stops
    signals['BuyStop'] = df['High'].rolling(window=entry_length, min_periods=1).max()
    signals['ShortStop'] = df['Low'].rolling(window=entry_length, min_periods=1).min()
    signals['CoverStop'] = df['High'].rolling(window=exit_length, min_periods=1).max()
    signals['SellStop'] = df['Low'].rolling(window=exit_length, min_periods=1).min()
    
    # Compute ATR
    atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_length).average_true_range()

    # Apply chosen smoothing type
    if atr_average_type == "simple":
        atr_smoothed = atr.rolling(window=atr_length, min_periods=1).mean()
    elif atr_average_type == "exponential":
        atr_smoothed = atr.ewm(span=atr_length, adjust=False).mean()
    elif atr_average_type == "wilder":
        atr_smoothed = atr.ewm(alpha=1/atr_length, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported ATR smoothing type '{atr_average_type}'")

    # Preinitialize signal column
    signals['donchian_signals'] = 0  # Default to 0 (neutral)

    # Apply ATR filter
    if atr_factor > 0:
        volatility_filter = atr_smoothed * atr_factor

        # Ensure proper alignment of boolean masks
        buy_condition = (df['High'] > signals['BuyStop']) & ((df['High'].shift(1) - volatility_filter) > 0)
        sell_condition = (df['Low'] < signals['ShortStop']) & ((df['Low'].shift(1) - volatility_filter) < 0)

        signals.loc[buy_condition, 'donchian_signals'] = 1
        signals.loc[sell_condition, 'donchian_signals'] = -1
    else:
        signals.loc[df['High'] > signals['BuyStop'], 'donchian_signals'] = 1
        signals.loc[df['Low'] < signals['ShortStop'], 'donchian_signals'] = -1

    # Apply ATR-based stop loss
    if atr_stop_factor > 0:
        atr_stop_loss = atr_smoothed * atr_stop_factor
        signals['BuyToClose'] = np.where(df['High'] > atr_stop_loss, 1, 0)
        signals['SellToClose'] = np.where(df['Low'] < atr_stop_loss, -1, 0)
    else:
        signals['BuyToClose'] = np.where(df['High'] > signals['CoverStop'], 1, 0)
        signals['SellToClose'] = np.where(df['Low'] < signals['SellStop'], -1, 0)

    # Drop unnecessary columns
    signals.drop(columns=['CoverStop', 'SellStop', 'BuyStop', 'ShortStop', 'BuyToClose', 'SellToClose'], inplace=True)

    return signals.fillna(0)


# %%
