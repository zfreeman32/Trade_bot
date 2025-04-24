
import pandas as pd
import numpy as np
from ta import momentum

# Membership Strategy Signals
def membership_strategy_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Rate of Change (ROC)
    signals['ROC'] = stock_df['Close'].pct_change(periods=5)  # 5-day ROC for momentum detection
    
    # Define conditions for signals
    signals['membership_signal'] = 'neutral'
    signals.loc[(signals['ROC'] > 0.02), 'membership_signal'] = 'long'  # Enter long if ROC is above 2%
    signals.loc[(signals['ROC'] < -0.02), 'membership_signal'] = 'short'  # Enter short if ROC is below -2%
    
    return signals[['membership_signal']]
