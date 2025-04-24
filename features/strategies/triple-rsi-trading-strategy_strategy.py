
import pandas as pd
import numpy as np
from ta import momentum, trend

# Triple RSI Trading Strategy
def triple_rsi_signals(stock_df, rsi_period=14, rsi_overbought=70, rsi_oversold=30, ma_window=200):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()

    # Calculate 200-day moving average
    signals['MA200'] = stock_df['Close'].rolling(window=ma_window).mean()

    # Generate signals
    signals['triple_rsi_signal'] = 'neutral'
    three_down_days = signals['RSI'].rolling(window=3).apply(lambda x: all(x[i] < rsi_oversold for i in range(3)), raw=True)

    signals.loc[
        (three_down_days) & (stock_df['Close'] < signals['MA200']) & (signals['RSI'] < rsi_oversold),
        'triple_rsi_signal'] = 'long'
    
    signals.loc[
        (signals['RSI'] > rsi_overbought) & (stock_df['Close'] > signals['MA200']),
        'triple_rsi_signal'] = 'short'

    return signals[['triple_rsi_signal', 'RSI', 'MA200']]
