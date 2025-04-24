
import pandas as pd
import numpy as np

# Ease Of Movement (EMV) Indicator Strategy
def emv_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Ease of Movement (EMV) indicator
    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']
    volume = stock_df['Volume']
    
    emv = (0.5 * (high - low) / volume) * 1000000
    emv_rolling = emv.rolling(window=window).mean()
    
    signals['EMV'] = emv_rolling
    signals['emv_signal'] = 'neutral'
    
    # Generate buy/sell signals based on EMV values
    signals.loc[(signals['EMV'] > 0) & (signals['EMV'].shift(1) <= 0), 'emv_signal'] = 'long'
    signals.loc[(signals['EMV'] < 0) & (signals['EMV'].shift(1) >= 0), 'emv_signal'] = 'short'
    
    return signals.drop(['EMV'], axis=1)
