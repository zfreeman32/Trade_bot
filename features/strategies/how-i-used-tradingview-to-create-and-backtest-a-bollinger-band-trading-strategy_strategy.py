
import pandas as pd
import numpy as np
from ta.volatility import BollingerBands

# Bollinger Bands Trading Strategy
def bollinger_band_signals(stock_df, window=20, std_dev=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Bollinger Bands
    bb = BollingerBands(close=stock_df['Close'], window=window, window_dev=std_dev)
    stock_df['BB_High'] = bb.bollinger_hband()
    stock_df['BB_Low'] = bb.bollinger_lband()
    stock_df['BB_Middle'] = bb.bollinger_mavg()
    
    # Initialize signals
    signals['bb_signal'] = 'neutral'
    
    # Generate buy/sell signals
    signals.loc[(stock_df['Close'] < stock_df['BB_Low']), 'bb_signal'] = 'long'  # Buy when price is below lower band
    signals.loc[(stock_df['Close'] > stock_df['BB_High']), 'bb_signal'] = 'short'  # Sell when price is above upper band
    
    return signals
