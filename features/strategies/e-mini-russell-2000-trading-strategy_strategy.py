
import pandas as pd
from ta import momentum

# E-Mini Russell 2000 Trading Strategy
def emini_russell_2000_signals(stock_df, window=14, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    signals['RSI'] = momentum.RSIIndicator(stock_df['Close'], window).rsi()
    signals['signal'] = 'neutral'
    
    signals.loc[(signals['RSI'] < oversold), 'signal'] = 'long'
    signals.loc[(signals['RSI'] > overbought), 'signal'] = 'short'
    
    return signals[['signal']]
