
import pandas as pd
import numpy as np

# Pair Trading Strategy in SPY and TLT
def spy_tlt_pair_trade_signals(spy_df, tlt_df, normal_drop=1.5, normal_rise=1.5):
    signals = pd.DataFrame(index=spy_df.index)
    
    # Calculate daily returns
    spy_returns = spy_df['Close'].pct_change() * 100  # Convert to percentage
    tlt_returns = tlt_df['Close'].pct_change() * 100  # Convert to percentage
    
    # Create conditions for the pair trade signals
    signals['spy_signal'] = np.where(spy_returns < -normal_drop, 'long', 'neutral')
    signals['tlt_signal'] = np.where(tlt_returns > normal_rise, 'short', 'neutral')
    
    # Combine signals
    signals['signal'] = np.where((signals['spy_signal'] == 'long') & (signals['tlt_signal'] == 'short'), 'trade', 'no_trade')
    
    return signals[['signal']]
