import pandas as pd
import numpy as np
from ta import trend

# SwingThree Strategy
def swingthree_signals(stock_df, sma_length=14, ema_length=50, tick_sizes=5):
    signals = pd.DataFrame(index=stock_df.index)
    sma_high = trend.sma_indicator(stock_df['High'], sma_length)
    sma_low = trend.sma_indicator(stock_df['Low'], sma_length)
    ema_close = trend.ema_indicator(stock_df['Close'], ema_length)
    signals['sma_high'] = sma_high
    signals['sma_low'] = sma_low
    signals['ema_close'] = ema_close
    signals['swingthree_signals_long_entry'] = (stock_df['High'] > sma_high + tick_sizes) & (stock_df['Close'].shift(1) > ema_close)
    signals['long_exit'] = stock_df['Low'] <= sma_low
    signals['swingthree_signals_short_entry'] = (stock_df['Low'] < sma_low - tick_sizes) & (stock_df['Close'].shift(1) < ema_close)
    signals['short_exit'] = stock_df['High'] >= sma_high
    signals.drop(['sma_high', 'sma_low','short_exit','ema_close','long_exit'], axis=1, inplace=True)

    return signals

