
import pandas as pd
from ta import trend

# GBPUSD Trading Strategy
def gbpusd_signals(data_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=data_df.index)
    # Calculate moving averages
    signals['short_mavg'] = data_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = data_df['Close'].rolling(window=long_window).mean()
    
    # Generate signals
    signals['signal'] = 0
    signals.loc[signals['short_mavg'] > signals['long_mavg'], 'signal'] = 1  # Long signal
    signals.loc[signals['short_mavg'] < signals['long_mavg'], 'signal'] = -1  # Short signal
    
    # Assign 'long', 'short', or 'neutral' based on the signal
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['signal'] == 1, 'trade_signal'] = 'long'
    signals.loc[signals['signal'] == -1, 'trade_signal'] = 'short'
    
    # Cleanup
    signals.drop(['short_mavg', 'long_mavg', 'signal'], axis=1, inplace=True)
    
    return signals
