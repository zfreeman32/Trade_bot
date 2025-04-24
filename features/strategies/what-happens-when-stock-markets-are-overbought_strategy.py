
import pandas as pd
from ta.momentum import RSIIndicator

# Overbought Market Trading Strategy
def overbought_market_signals(stock_df, rsi_period=14, overbought_threshold=70):
    signals = pd.DataFrame(index=stock_df.index)
    rsi = RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()
    signals['signal'] = 'neutral'
    
    # Generate signals based on RSI
    signals.loc[signals['RSI'] > overbought_threshold, 'signal'] = 'short'
    signals.loc[signals['RSI'] <= overbought_threshold, 'signal'] = 'neutral'
    
    return signals
