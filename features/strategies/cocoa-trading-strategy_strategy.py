
import pandas as pd
import numpy as np
from ta import momentum, trend

# Cocoa Trading Strategy
def cocoa_trading_signals(cocoa_df, short_window=14, long_window=50):
    signals = pd.DataFrame(index=cocoa_df.index)
    signals['Close'] = cocoa_df['Close']
    
    # Calculate short and long moving averages
    signals['Short_MA'] = signals['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = signals['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'signal'] = 1  # Buy signal
    signals.loc[signals['Short_MA'] < signals['Long_MA'], 'signal'] = -1  # Sell signal
    
    # Convert signal to trading decisions
    signals['cocoa_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'cocoa_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'cocoa_signal'] = 'short'

    # Clean the DataFrame to contain only relevant columns
    signals = signals[['cocoa_signal']]
    
    return signals
