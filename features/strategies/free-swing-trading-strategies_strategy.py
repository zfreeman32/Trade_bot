
import pandas as pd
import numpy as np
from ta import momentum, trend

# RSI and Moving Average Crossover Strategy
def rsi_ma_crossover_signals(stock_df, rsi_window=14, ma_short_window=20, ma_long_window=50, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi.rsi()
    
    # Calculate moving averages
    signals['MA_Short'] = stock_df['Close'].rolling(window=ma_short_window).mean()
    signals['MA_Long'] = stock_df['Close'].rolling(window=ma_long_window).mean()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['MA_Short'] > signals['MA_Long']) & (signals['RSI'] < rsi_oversold), 'signal'] = 'long'
    signals.loc[(signals['MA_Short'] < signals['MA_Long']) & (signals['RSI'] > rsi_overbought), 'signal'] = 'short'
    
    # Clean up DataFrame
    signals.drop(['RSI', 'MA_Short', 'MA_Long'], axis=1, inplace=True)
    
    return signals
