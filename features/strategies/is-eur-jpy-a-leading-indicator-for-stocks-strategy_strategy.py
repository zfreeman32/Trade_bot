
import pandas as pd
import numpy as np
from ta import momentum

# Momentum and Momentum Reversal Strategy
def momentum_reversal_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    # Calculate momentum using the Rate of Change (ROC)
    momentum_indicator = stock_df['Close'].pct_change(periods=window) * 100
    signals['momentum'] = momentum_indicator
    
    # Generate buy and sell signals based on momentum reversal criteria
    signals['signal'] = 'neutral'
    signals.loc[(signals['momentum'] > 0) & (signals['momentum'].shift(1) <= 0), 'signal'] = 'long'
    signals.loc[(signals['momentum'] < 0) & (signals['momentum'].shift(1) >= 0), 'signal'] = 'short'
    
    # Optional: Remove the momentum column for cleaner output
    signals.drop(['momentum'], axis=1, inplace=True)
    
    return signals
