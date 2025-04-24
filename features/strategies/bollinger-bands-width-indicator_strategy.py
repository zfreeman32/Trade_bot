
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands

# Bollinger Bands Width Trading Strategy
def bollinger_bands_width_signals(stock_df, window=20, num_std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=stock_df['Close'], window=window, window_dev=num_std_dev)
    
    # Calculate the Bollinger Bands Width
    signals['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / stock_df['Close'] * 100
    
    # Identify signals based on Bollinger Bands Width
    signals['bb_signal'] = 'neutral'
    signals.loc[(signals['BB_Width'] > signals['BB_Width'].rolling(window).mean()), 'bb_signal'] = 'long'
    signals.loc[(signals['BB_Width'] < signals['BB_Width'].rolling(window).mean()), 'bb_signal'] = 'short'
    
    return signals[['bb_signal']]
