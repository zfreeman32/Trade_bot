
import pandas as pd
import numpy as np

# CAD/CHF Forex Trading Strategy
def cad_chf_signals(forex_df, lookback_period=90, threshold_up=0.0005, threshold_down=-0.0005):
    signals = pd.DataFrame(index=forex_df.index)
    signals['Close'] = forex_df['Close']

    # Calculate price change over specified lookback period
    price_change = forex_df['Close'].diff(lookback_period)
    
    # Generate signals based on price change thresholds
    signals['signal'] = 'neutral'
    signals.loc[price_change > threshold_up, 'signal'] = 'long'
    signals.loc[price_change < threshold_down, 'signal'] = 'short'
    
    return signals[['signal']]
