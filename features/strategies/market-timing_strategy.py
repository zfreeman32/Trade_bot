
import pandas as pd
from ta import trend

# 200-Day Moving Average Strategy
def market_timing_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['200_MA'] = stock_df['Close'].rolling(window=200).mean()

    signals['signal'] = 'neutral'
    signals.loc[stock_df['Close'] > signals['200_MA'], 'signal'] = 'long'
    signals.loc[stock_df['Close'] < signals['200_MA'], 'signal'] = 'short'
    
    return signals[['signal']]
