
import pandas as pd
import numpy as np

# Price Rate of Change (ROC) Strategy
def roc_signals(stock_df, period=14, overbought=20, oversold=-20):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Rate of Change
    roc = stock_df['Close'].pct_change(periods=period) * 100  # Convert to percentage
    signals['ROC'] = roc

    # Initialize signals
    signals['roc_signal'] = 'neutral'
    
    # Generate buy/sell signals based on ROC
    signals.loc[(signals['ROC'] > overbought) & (signals['ROC'].shift(1) <= overbought), 'roc_signal'] = 'short'
    signals.loc[(signals['ROC'] < oversold) & (signals['ROC'].shift(1) >= oversold), 'roc_signal'] = 'long'
    
    return signals.drop(['ROC'], axis=1)
