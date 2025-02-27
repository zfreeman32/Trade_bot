import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib

# SwingThree Strategy
def swingthree_signals(stock_df, sma_length=14, ema_length=50, tick_sizes=5):
    signals = pd.DataFrame(index=stock_df.index)
    sma_high = trend.sma_indicator(stock_df['High'], sma_length)
    sma_low = trend.sma_indicator(stock_df['Low'], sma_length)
    ema_close = trend.ema_indicator(stock_df['Close'], ema_length)
    
    buy_signal_col = 'swingthree_buy_signal'
    sell_signal_col = 'swingthree_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    signals.loc[(stock_df['High'] > sma_high + tick_sizes) & (stock_df['Close'].shift(1) > ema_close), buy_signal_col] = 1
    signals.loc[(stock_df['Low'] < sma_low - tick_sizes) & (stock_df['Close'].shift(1) < ema_close), sell_signal_col] = 1
    
    return signals
