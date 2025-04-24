
import pandas as pd
import numpy as np
from ta import momentum

# Moving Average Convergence Divergence (MACD) Strategy
def macd_signals(stock_df, short_window=12, long_window=26, signal_window=9):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate MACD
    exp1 = stock_df['Close'].ewm(span=short_window, adjust=False).mean()
    exp2 = stock_df['Close'].ewm(span=long_window, adjust=False).mean()
    macd = exp1 - exp2
    signals['MACD'] = macd
    
    # Calculate Signal Line
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    signals['Signal'] = signal
    
    # Generate trading signals
    signals['macd_signal'] = 'neutral'
    signals.loc[(signals['MACD'] > signals['Signal']) & (signals['MACD'].shift(1) <= signals['Signal'].shift(1)), 'macd_signal'] = 'long'
    signals.loc[(signals['MACD'] < signals['Signal']) & (signals['MACD'].shift(1) >= signals['Signal'].shift(1)), 'macd_signal'] = 'short'
    
    return signals[['macd_signal']]
