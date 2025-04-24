
import pandas as pd
import numpy as np
from ta import momentum

# Momentum Trading Strategy
def momentum_trading_signals(stock_df, window=14, threshold=0.02):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the momentum
    momentum_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = momentum_indicator.rsi()
    
    # Generate signals
    signals['signal'] = 'neutral'
    signals.loc[(signals['RSI'] > 70), 'signal'] = 'short'  # Overbought condition
    signals.loc[(signals['RSI'] < 30), 'signal'] = 'long'   # Oversold condition
    
    # Shift signals to avoid lookahead bias
    signals['signal'] = signals['signal'].shift(1)
    
    return signals[['signal']]
