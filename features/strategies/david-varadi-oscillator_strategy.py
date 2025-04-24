
import pandas as pd
import numpy as np

# David Varadi Oscillator (DVO) Strategy
def dvo_signals(stock_df, lookback_period=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the median price over the lookback period
    median_price = stock_df['Close'].rolling(window=lookback_period).median()

    # Calculate the detrended prices
    detrended_prices = stock_df['Close'] / median_price

    # Calculate the rolling percent rank of detrended prices
    rank = detrended_prices.rolling(window=lookback_period).apply(lambda x: np.percentile(x, 100), raw=True)
    dvo = 100 * (detrended_prices - rank) / rank
    
    # Generate signals
    signals['DVO'] = dvo
    signals['dvo_signal'] = 'neutral'
    signals.loc[(signals['DVO'] > 0) & (signals['DVO'].shift(1) <= 0), 'dvo_signal'] = 'long'
    signals.loc[(signals['DVO'] < 0) & (signals['DVO'].shift(1) >= 0), 'dvo_signal'] = 'short'
    signals.drop(['DVO'], axis=1, inplace=True)
    
    return signals
