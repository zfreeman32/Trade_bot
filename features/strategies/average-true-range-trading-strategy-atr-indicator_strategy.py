
import pandas as pd
from ta import momentum, trend

# Moving Average Convergence Divergence (MACD) Strategy
def macd_signals(stock_df, window_slow=26, window_fast=12, signal_window=9):
    signals = pd.DataFrame(index=stock_df.index)
    macd = trend.MACD(stock_df['Close'], window_slow=window_slow, window_fast=window_fast, window_sign=signal_window)
    signals['MACD'] = macd.macd()
    signals['MACD_Signal'] = macd.macd_signal()
    signals['macd_signal'] = 'neutral'
    
    signals.loc[(signals['MACD'] > signals['MACD_Signal']) & (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'long'
    signals.loc[(signals['MACD'] < signals['MACD_Signal']) & (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'short'
    
    return signals
