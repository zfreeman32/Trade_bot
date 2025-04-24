
import pandas as pd
import numpy as np
from ta import momentum, trend

# Energy Sector Trading Strategy
def energy_sector_signals(stock_df, long_window=50, short_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    
    # Calculate moving averages
    signals['Long_MA'] = stock_df['Close'].rolling(window=long_window).mean()
    signals['Short_MA'] = stock_df['Close'].rolling(window=short_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['Short_MA'][short_window:] > signals['Long_MA'][short_window:], 1, 0)
    signals['positions'] = signals['signal'].diff()
    
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['positions'] == 1, 'trade_signal'] = 'long'  # Buy signal
    signals.loc[signals['positions'] == -1, 'trade_signal'] = 'short'  # Sell signal
    
    return signals[['Close', 'Long_MA', 'Short_MA', 'trade_signal']]
