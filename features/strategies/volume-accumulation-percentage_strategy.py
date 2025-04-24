
import pandas as pd
import numpy as np

# Volume Accumulation Percentage (VAP) Strategy
def vap_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate VAP
    price = stock_df['Close']
    volume = stock_df['Volume']
    positive_volume = np.where(price.diff() >= 0, volume, 0)
    negative_volume = np.where(price.diff() < 0, volume, 0)
    money_flow = positive_volume - negative_volume
    
    vap = (money_flow.rolling(window).sum() / volume.rolling(window).sum()) * 100
    
    # Generate trading signals based on VAP
    signals['VAP'] = vap
    signals['vap_signal'] = 'neutral'
    signals.loc[(signals['VAP'] > 0), 'vap_signal'] = 'long'
    signals.loc[(signals['VAP'] < 0), 'vap_signal'] = 'short'
    
    return signals[['vap_signal']]
