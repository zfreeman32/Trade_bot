
import pandas as pd
import numpy as np

# Runaway Gap Trading Strategy
def runaway_gap_signals(stock_df, gap_threshold=0.02, volume_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    signals['Volume'] = stock_df['Volume']
    
    # Identify if there is a gap
    signals['Gap'] = signals['Close'].shift(1) * (1 + gap_threshold) < signals['Close']
    
    # Identify if the current volume is above the average volume * volume_multiplier
    volume_avg = signals['Volume'].rolling(window=20).mean()
    signals['High_Volume'] = signals['Volume'] > (volume_avg * volume_multiplier)
    
    # Generate signals based on gap and volume conditions
    signals['runaway_gap_signal'] = 'neutral'
    signals.loc[signals['Gap'] & signals['High_Volume'], 'runaway_gap_signal'] = 'long'
    signals.loc[signals['Gap'].shift(1) & signals['High_Volume'].shift(1), 'runaway_gap_signal'] = 'short'
    
    return signals[['runaway_gap_signal']]
