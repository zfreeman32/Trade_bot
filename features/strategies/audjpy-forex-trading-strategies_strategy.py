
import pandas as pd
import numpy as np

# AUDJPY Trading Strategy
def audjpy_signals(forex_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=forex_df.index)
    signals['Close'] = forex_df['Close']
    
    # Calculate moving averages
    signals['short_mavg'] = forex_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = forex_df['Close'].rolling(window=long_window).mean()
    
    # Create signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1, 0)
    
    # Generate a position from the signals
    signals['position'] = signals['signal'].diff()
    
    # Define trading signals
    signals['audjpy_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'audjpy_signal'] = 'long'   # Buy signal
    signals.loc[signals['position'] == -1, 'audjpy_signal'] = 'short'  # Sell signal
    
    # Drop intermediate columns
    signals.drop(['short_mavg', 'long_mavg', 'signal', 'position'], axis=1, inplace=True)
    
    return signals
