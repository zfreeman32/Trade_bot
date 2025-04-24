
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# Membership Strategy Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Example of using moving averages for signal generation
    short_window = 20
    long_window = 50
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    
    # Generating signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(
        signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0
    )
    
    signals['positions'] = signals['signal'].diff()
    
    signals['membership_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'membership_signal'] = 'long'
    signals.loc[signals['positions'] == -1, 'membership_signal'] = 'short'
    
    return signals[['membership_signal']]
