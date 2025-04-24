
import pandas as pd
from ta.volume import VolumeWeightedAveragePrice

# Volume Oscillator Strategy
def volume_oscillator_signals(stock_df, short_window=5, long_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the short-term and long-term moving averages of volume
    volume_short_ma = stock_df['Volume'].rolling(window=short_window).mean()
    volume_long_ma = stock_df['Volume'].rolling(window=long_window).mean()
    
    # Calculate the Volume Oscillator
    signals['Volume_Oscillator'] = (volume_short_ma - volume_long_ma) / volume_long_ma * 100
    
    # Generate signals based on Volume Oscillator
    signals['signal'] = 'neutral'
    signals.loc[(signals['Volume_Oscillator'] > 0) & (signals['Volume_Oscillator'].shift(1) <= 0), 'signal'] = 'long'
    signals.loc[(signals['Volume_Oscillator'] < 0) & (signals['Volume_Oscillator'].shift(1) >= 0), 'signal'] = 'short'
    
    return signals[['signal']]
