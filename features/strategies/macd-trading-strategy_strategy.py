
import pandas as pd
from ta import trend

# MACD Trading Strategy
def macd_signals(stock_df, short_window=12, long_window=26, signal_window=9):
    signals = pd.DataFrame(index=stock_df.index)
    macd = trend.MACD(stock_df['Close'], window_slow=long_window, window_fast=short_window, window_sign=signal_window)
    
    signals['MACD'] = macd.macd()
    signals['MACD_Signal'] = macd.macd_signal()
    
    signals['macd_signal'] = 'neutral'
    signals.loc[(signals['MACD'] > signals['MACD_Signal']) & (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'long'
    signals.loc[(signals['MACD'] < signals['MACD_Signal']) & (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1)), 'macd_signal'] = 'short'
    
    signals.drop(['MACD', 'MACD_Signal'], axis=1, inplace=True)
    return signals
