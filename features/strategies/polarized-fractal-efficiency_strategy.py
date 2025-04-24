
import pandas as pd
import numpy as np

# Polarized Fractal Efficiency (PFE) Strategy
def pfe_signals(stock_df, length=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate PFE
    high = stock_df['High'].rolling(window=length).max()
    low = stock_df['Low'].rolling(window=length).min()
    close = stock_df['Close']
    pfe = ((close - low) / (high - low)) * 100 - 50
    signals['PFE'] = pfe
    
    # Generate trading signals
    signals['pfe_signal'] = 'neutral'
    signals.loc[(signals['PFE'] > 0) & (signals['PFE'].shift(1) <= 0), 'pfe_signal'] = 'long'
    signals.loc[(signals['PFE'] < 0) & (signals['PFE'].shift(1) >= 0), 'pfe_signal'] = 'short'
    
    signals.drop(['PFE'], axis=1, inplace=True)
    return signals
