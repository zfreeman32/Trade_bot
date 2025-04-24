
import pandas as pd

# All-Time High Trading Strategy
def all_time_high_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['previous_close'] = stock_df['Close'].shift(1)
    signals['all_time_high'] = stock_df['Close'].cummax()
    
    # Generate signals based on all-time high
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['all_time_high'].shift(1)) & (stock_df['Close'] >= stock_df['previous_close']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['all_time_high'].shift(1)], 'signal'] = 'short'
    
    # Dropping unnecessary intermediate calculation
    signals.drop(['previous_close', 'all_time_high'], axis=1, inplace=True)
    return signals
