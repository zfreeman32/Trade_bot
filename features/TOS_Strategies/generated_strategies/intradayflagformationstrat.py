import pandas as pd
import numpy as np
from ta.volatility import AverageTrueRange
from ta.trend import SMAIndicator

def flag_formation_signals(ohlcv_df, atr_window=14, flag_len=15, flag_atr_multiplier=2.5, 
                           pole_len=23, pole_atr_multiplier=5.5, uptrend_len=70, 
                           flag_gap=50, atr_percent_change=0.05):
    
    # ATR
    atr = AverageTrueRange(ohlcv_df['High'], ohlcv_df['Low'], ohlcv_df['Close'], atr_window).average_true_range()
    
    # SMA for uptrend
    sma = SMAIndicator(ohlcv_df['Close'], uptrend_len).sma_indicator()
    
    # Identify poles
    ohlcv_df['Pole'] = ((ohlcv_df['High'] - ohlcv_df['Low']) >= pole_atr_multiplier * atr).astype(int)
    
    # Identify flags
    ohlcv_df['flag_formation_signals'] = 0  # Initialize the column

    for i in range(pole_len, len(ohlcv_df)):
        # Use .iloc with .values[] to avoid FutureWarning
        high_range = ohlcv_df['High'].iloc[i-flag_len:i].values
        low_range = ohlcv_df['Low'].iloc[i-flag_len:i].values
        atr_value = atr.iloc[i]  # Ensure atr is accessed using .iloc

        if ohlcv_df['Pole'].iloc[i-pole_len:i].sum() == 0 and (high_range - low_range).max() <= flag_atr_multiplier * atr_value:
            ohlcv_df.loc[ohlcv_df.index[i], 'flag_formation_signals'] = 1

    # Filter for uptrend and minimum gap between flags
    ohlcv_df['Uptrend'] = ohlcv_df['Close'] > sma
    ohlcv_df['flag_formation_signals'] = ohlcv_df['flag_formation_signals'] * ohlcv_df['Uptrend']
    
    flags = ohlcv_df[ohlcv_df['flag_formation_signals'] == 1].index
    valid_flags = [flag for i, flag in enumerate(flags) if i == 0 or (flags[i] - flags[i-1]) > pd.Timedelta(days=flag_gap)]
    
    ohlcv_df['flag_formation_signals'] = ohlcv_df.index.isin(valid_flags).astype(int)

    # Filter for ATR percent change
    ohlcv_df['ATR_Change'] = atr.pct_change() >= atr_percent_change
    ohlcv_df['flag_formation_signals'] = ohlcv_df['flag_formation_signals'] * ohlcv_df['ATR_Change']

    return ohlcv_df['flag_formation_signals'].replace({1: 'long', 0: 'neutral'})


