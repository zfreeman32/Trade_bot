
import pandas as pd
import numpy as np
from ta import momentum

# RSI and Moving Average Crossover Strategy
def rsi_mac_signals(stock_df, rsi_window=14, ma_short_window=10, ma_long_window=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi.rsi()
    
    # Calculate Moving Averages
    signals['MA_Short'] = stock_df['Close'].rolling(window=ma_short_window).mean()
    signals['MA_Long'] = stock_df['Close'].rolling(window=ma_long_window).mean()

    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['RSI'] < 30) & (signals['MA_Short'] > signals['MA_Long']), 'signal'] = 'long'
    signals.loc[(signals['RSI'] > 70) & (signals['MA_Short'] < signals['MA_Long']), 'signal'] = 'short'
    
    # Drop intermediate calculations
    signals.drop(['RSI', 'MA_Short', 'MA_Long'], axis=1, inplace=True)
    return signals
