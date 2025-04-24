
import pandas as pd
import numpy as np

# Real Estate Sector Trading Strategy
def real_estate_sector_signals(stock_df, short_window=20, long_window=50):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window, min_periods=1).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window, min_periods=1).mean()
    
    signals['signal'] = 0
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'signal'] = 1  # Long signal
    signals.loc[signals['Short_MA'] < signals['Long_MA'], 'signal'] = -1  # Short signal
    
    signals['strategy'] = signals['signal'].diff()
    
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['strategy'] == 2, 'trading_signal'] = 'long'  # Buy signal
    signals.loc[signals['strategy'] == -2, 'trading_signal'] = 'short'  # Sell signal
    
    signals.drop(['Short_MA', 'Long_MA', 'signal', 'strategy'], axis=1, inplace=True)
    return signals
