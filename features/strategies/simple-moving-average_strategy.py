
import pandas as pd
import numpy as np
from ta import momentum

# Momentum-Based Trading Strategy
def momentum_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Momentum'] = momentum.MACD(stock_df['Close']).macd()
    signals['signal'] = 'neutral'
    signals.loc[(signals['Momentum'] > 0) & (signals['Momentum'].shift(1) <= 0), 'signal'] = 'long'
    signals.loc[(signals['Momentum'] < 0) & (signals['Momentum'].shift(1) >= 0), 'signal'] = 'short'
    return signals[['signal']]
