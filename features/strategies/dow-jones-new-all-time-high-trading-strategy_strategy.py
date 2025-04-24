
import pandas as pd
import numpy as np

# Quantified Strategies Trading Strategy
def quantified_strategies_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Define quantifiable rules for entry and exit based on patterns and anomalies
    signals['returns'] = stock_df['Close'].pct_change()
    signals['moving_average'] = stock_df['Close'].rolling(window=20).mean()
    
    # Define long and short conditions based on the described anomalies
    signals['long_condition'] = (signals['returns'] > 0) & (stock_df['Close'] > signals['moving_average'])
    signals['short_condition'] = (signals['returns'] < 0) & (stock_df['Close'] < signals['moving_average'])
    
    # Generate trading signals
    signals['strategy_signal'] = 'neutral'
    signals.loc[signals['long_condition'], 'strategy_signal'] = 'long'
    signals.loc[signals['short_condition'], 'strategy_signal'] = 'short'

    # Clean up the DataFrame
    signals.drop(['returns', 'moving_average', 'long_condition', 'short_condition'], axis=1, inplace=True)
    return signals
