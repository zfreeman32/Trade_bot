
import pandas as pd
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator

# 50 EMA Trading Strategy
def ema_50_signals(stock_df, rsi_window=14, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate 50-day EMA
    ema_indicator = EMAIndicator(close=stock_df['Close'], window=50)
    stock_df['EMA50'] = ema_indicator.ema_indicator()
    
    # Calculate RSI
    rsi_indicator = RSIIndicator(close=stock_df['Close'], window=rsi_window)
    stock_df['RSI'] = rsi_indicator.rsi()
    
    # Initialize signals
    signals['ema_signal'] = 'neutral'
    
    # Conditions for buying and selling
    signals.loc[(stock_df['Close'] > stock_df['EMA50']) & (stock_df['RSI'] < rsi_oversold), 'ema_signal'] = 'long'
    signals.loc[(stock_df['Close'] < stock_df['EMA50']) & (stock_df['RSI'] > rsi_overbought), 'ema_signal'] = 'short'
    
    return signals
