
import pandas as pd
from ta import momentum

# Relative Strength Index (RSI) Strategy
def rsi_signals(stock_df, rsi_period=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()
    signals['rsi_signal'] = 'neutral'
    signals.loc[(signals['RSI'] < oversold), 'rsi_signal'] = 'long'
    signals.loc[(signals['RSI'] > overbought), 'rsi_signal'] = 'short'
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals
