
import pandas as pd
import numpy as np
from ta import momentum, volatility

# Dynamic Zone RSI Strategy
def dynamic_zone_rsi_signals(stock_df, rsi_period=14, bb_period=20, bb_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_period).rsi()
    
    # Calculate the Bollinger Bands for RSI
    rsi_bollinger = volatility.BollingerBands(rsi, window=bb_period, window_dev=bb_std_dev)
    
    upper_band = rsi_bollinger.bollinger_hband()
    lower_band = rsi_bollinger.bollinger_lband()
    
    signals['RSI'] = rsi
    signals['Upper_Band'] = upper_band
    signals['Lower_Band'] = lower_band
    signals['rsi_signal'] = 'neutral'
    
    signals.loc[(signals['RSI'] > signals['Upper_Band']), 'rsi_signal'] = 'short'
    signals.loc[(signals['RSI'] < signals['Lower_Band']), 'rsi_signal'] = 'long'
    
    return signals[['rsi_signal']]
