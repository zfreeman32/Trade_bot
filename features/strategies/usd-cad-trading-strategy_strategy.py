
import pandas as pd
import numpy as np

# USDCAD Forex Trading Strategy
def usdcad_trading_signals(forex_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = forex_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = forex_df['Close'].rolling(window=long_window).mean()
    
    # Initialize signals
    signals['signal'] = 0.0
    
    # Generate signals based on moving average crossover
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['position'] = signals['signal'].diff()

    # Creating our strategy signals
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short'
    
    return signals[['trading_signal']]
