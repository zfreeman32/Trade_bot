
import pandas as pd
import numpy as np
from ta import momentum

# Divergence Trading Strategy
def divergence_signals(stock_df, indicator: str = 'RSI', threshold: float = 30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the chosen indicator (e.g., RSI)
    if indicator == 'RSI':
        rsi = momentum.RSIIndicator(stock_df['Close'])
        signals['indicator'] = rsi.rsi()
    else:
        raise ValueError("Currently only RSI is supported as an indicator.")
    
    # Initialize the signal column
    signals['signal'] = 'neutral'
    
    # Identifying bullish divergence
    bullish_divergence = (stock_df['Close'].rolling(window=5).max().shift(1) > stock_df['Close'].rolling(window=5).max()) & \
                         (signals['indicator'].rolling(window=5).min().shift(1) < signals['indicator'].rolling(window=5).min())
    
    # Identifying bearish divergence
    bearish_divergence = (stock_df['Close'].rolling(window=5).min().shift(1) < stock_df['Close'].rolling(window=5).min()) & \
                         (signals['indicator'].rolling(window=5).max().shift(1) > signals['indicator'].rolling(window=5).max())
    
    # Generating signals based on divergence
    signals.loc[bullish_divergence, 'signal'] = 'long'
    signals.loc[bearish_divergence, 'signal'] = 'short'
    
    return signals
