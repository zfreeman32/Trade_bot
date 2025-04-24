
import pandas as pd
import numpy as np

# Micro E-mini S&P 500 Trading Strategy
def micro_e_mini_sp500_signals(price_df, moving_average_window=20, threshold=0.02):
    signals = pd.DataFrame(index=price_df.index)
    signals['Close'] = price_df['Close']
    
    # Calculate moving average
    signals['Moving_Average'] = signals['Close'].rolling(window=moving_average_window).mean()
    
    # Generate signals based on rules
    signals['Signal'] = 'neutral'
    signals.loc[(signals['Close'] > signals['Moving_Average'] * (1 + threshold)), 'Signal'] = 'long'
    signals.loc[(signals['Close'] < signals['Moving_Average'] * (1 - threshold)), 'Signal'] = 'short'
    
    return signals[['Close', 'Moving_Average', 'Signal']]
