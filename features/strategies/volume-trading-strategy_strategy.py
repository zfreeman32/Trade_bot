
import pandas as pd
import numpy as np

# Volume Trading Strategy
def volume_trading_signals(stock_df, volume_multiplier=1.5):
    signals = pd.DataFrame(index=stock_df.index)
    signals['volume'] = stock_df['Volume']
    signals['avg_volume'] = stock_df['Volume'].rolling(window=20).mean()

    # Generate signals based on volume
    signals['signal'] = 'neutral'
    signals.loc[(signals['volume'] > signals['avg_volume'] * volume_multiplier), 'signal'] = 'long'
    signals.loc[(signals['volume'] < signals['avg_volume'] / volume_multiplier), 'signal'] = 'short'

    return signals[['signal']]
