
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator

# Larry Connors' R3 Strategy
def r3_strategy_signals(stock_df, rsi_period=2, rsi_overbought=70, rsi_oversold=30, close_below_entry=False):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate RSI
    rsi = RSIIndicator(stock_df['Close'], window=rsi_period)
    signals['RSI'] = rsi.rsi()
    
    # Initialize signal column
    signals['r3_signal'] = 'neutral'
    
    # Generate buy signals
    buy_condition = (signals['RSI'] < rsi_oversold)
    signals.loc[buy_condition, 'r3_signal'] = 'long'
    
    # Handle closing below entry point
    if close_below_entry:
        additional_buy_condition = (signals['r3_signal'].shift(1) == 'long') & (stock_df['Close'] < stock_df['Close'].shift(1))
        signals.loc[additional_buy_condition, 'r3_signal'] = 'long'
    
    return signals.drop(['RSI'], axis=1)
