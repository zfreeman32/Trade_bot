
import pandas as pd
import numpy as np

# Copper Trading Strategy
def copper_trading_signals(copper_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=copper_df.index)
    
    # Calculate moving averages
    signals['short_ma'] = copper_df['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = copper_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1, 0)   
    signals['position'] = signals['signal'].diff()
    
    signals['trading_signal'] = 'neutral'
    signals.loc[signals['position'] == 1, 'trading_signal'] = 'long'      # Buy signal
    signals.loc[signals['position'] == -1, 'trading_signal'] = 'short'    # Sell signal
    signals.drop(['short_ma', 'long_ma', 'signal', 'position'], axis=1, inplace=True)

    return signals
