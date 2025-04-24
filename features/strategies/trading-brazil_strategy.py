
import pandas as pd
import numpy as np
from ta import momentum

# Day Trading and Swing Trading Signals Strategy for Brazil
def brazil_day_swing_signals(stock_df, short_window=5, long_window=20, atr_window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Calculate Average True Range (ATR) for volatility 
    high_low = stock_df['High'] - stock_df['Low']
    high_close = np.abs(stock_df['High'] - stock_df['Close'].shift(1))
    low_close = np.abs(stock_df['Low'] - stock_df['Close'].shift(1))
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
    signals['ATR'] = true_range.rolling(window=atr_window).mean()
    
    # Generate Signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['short_mavg'] > signals['long_mavg']) & (stock_df['Close'] > signals['short_mavg'] + signals['ATR']), 'signal'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']) & (stock_df['Close'] < signals['short_mavg'] - signals['ATR']), 'signal'] = 'short'
    
    return signals[['signal']]
