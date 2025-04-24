
import pandas as pd
from ta import momentum

# Momentum Reversal Trading Strategy
def momentum_reversal_signals(stock_df, lookback_period=14, threshold=0.05):
    signals = pd.DataFrame(index=stock_df.index)
    signals['momentum'] = stock_df['Close'].pct_change(periods=lookback_period)
    
    signals['signal'] = 'neutral'
    
    # Buy signal
    signals.loc[(signals['momentum'] > threshold), 'signal'] = 'long'
    # Sell signal
    signals.loc[(signals['momentum'] < -threshold), 'signal'] = 'short'
    
    return signals[['signal']]
