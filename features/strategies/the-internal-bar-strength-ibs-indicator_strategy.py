
import pandas as pd
import numpy as np
from ta import momentum, trend

# Breakout Trading Strategy
def breakout_trading_signals(stock_df, breakout_window=20, volatility_window=10):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate rolling maximum and minimum for breakout levels
    signals['high'] = stock_df['High'].rolling(window=breakout_window).max()
    signals['low'] = stock_df['Low'].rolling(window=breakout_window).min()
    
    # Calculate Average True Range (ATR) for volatility
    signals['atr'] = stock_df['Close'].rolling(window=volatility_window).apply(lambda x: np.max(x) - np.min(x))
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['high'].shift(1)), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['low'].shift(1)), 'signal'] = 'short'
    
    # Clean up the DataFrame
    signals.drop(['high', 'low', 'atr'], axis=1, inplace=True)
    
    return signals
