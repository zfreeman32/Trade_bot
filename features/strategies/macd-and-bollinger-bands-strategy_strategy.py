
import pandas as pd
import numpy as np
from ta import momentum, volatility

# MACD and Bollinger Bands Strategy
def macd_bollinger_signals(stock_df, bb_window=20, bb_std=2, macd_slow=26, macd_fast=12, macd_signal=9):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Bollinger Bands
    stock_df['MA'] = stock_df['Close'].rolling(window=bb_window).mean()
    stock_df['STD'] = stock_df['Close'].rolling(window=bb_window).std()
    stock_df['Upper_Band'] = stock_df['MA'] + (stock_df['STD'] * bb_std)
    stock_df['Lower_Band'] = stock_df['MA'] - (stock_df['STD'] * bb_std)

    # Calculate MACD
    macd_indicator = momentum.MACD(stock_df['Close'], n_slow=macd_slow, n_fast=macd_fast, n_sign=macd_signal)
    stock_df['MACD'] = macd_indicator.macd()
    stock_df['Signal_Line'] = macd_indicator.macd_signal()

    # Generate signals
    signals['macd_signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > stock_df['Upper_Band']) & (stock_df['MACD'] > stock_df['Signal_Line']), 'macd_signal'] = 'long'
    signals.loc[(stock_df['Close'] < stock_df['Lower_Band']) & (stock_df['MACD'] < stock_df['Signal_Line']), 'macd_signal'] = 'short'
    
    return signals
