
import pandas as pd
import numpy as np
from ta import momentum

# 200 Trading Strategies: Momentum Strategy
def momentum_strategy_signals(stock_df, short_window=10, long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate price momentum using rate of change
    signals['momentum'] = stock_df['Close'].pct_change(periods=short_window)
    signals['long_sma'] = stock_df['Close'].rolling(window=long_window).mean()
    
    signals['signal'] = 'neutral'
    signals.loc[(signals['momentum'] > 0) & (stock_df['Close'] > signals['long_sma']), 'signal'] = 'long'
    signals.loc[(signals['momentum'] < 0) & (stock_df['Close'] < signals['long_sma']), 'signal'] = 'short'
    
    return signals[['signal']]
