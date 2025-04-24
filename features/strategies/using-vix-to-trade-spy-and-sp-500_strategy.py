
import pandas as pd
import numpy as np
from ta import volatility

# VIX-Based Trading Strategy for SPY
def vix_spy_trading_signals(vix_df, spy_df, window_length=20):
    signals = pd.DataFrame(index=spy_df.index)
    
    # Calculate Bollinger Bands for VIX
    vix_df['MA'] = vix_df['Close'].rolling(window=window_length).mean()
    vix_df['Upper_Band'] = vix_df['MA'] + (vix_df['Close'].rolling(window=window_length).std() * 2)
    
    # Create signals based on VIX crossing upper Bollinger Band
    signals['vix_signal'] = np.where(vix_df['Close'] > vix_df['Upper_Band'], 'buy', 'neutral')
    
    # Look for two consecutive up days in SPY after VIX signal
    signals['spy_returns'] = spy_df['Close'].pct_change()
    signals['sp_days_up'] = (signals['spy_returns'] > 0).astype(int).rolling(window=2).sum()
    
    # Create final signals
    signals['final_signal'] = 'neutral'
    signals.loc[(signals['vix_signal'] == 'buy') & (signals['sp_days_up'] == 2), 'final_signal'] = 'long'
    
    return signals[['final_signal']]
