
import pandas as pd
from ta.momentum import RSIIndicator

# SPY RSI Trading Strategy
def spy_rsi_signals(stock_df, rsi_period=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()
    signals['rsi_signal'] = 'neutral'
    
    signals.loc[(signals['RSI'] < oversold), 'rsi_signal'] = 'long'
    signals.loc[(signals['RSI'] > overbought), 'rsi_signal'] = 'short'
    
    return signals[['rsi_signal']]
