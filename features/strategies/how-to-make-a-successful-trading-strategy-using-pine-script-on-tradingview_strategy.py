
import pandas as pd
import numpy as np
from ta import momentum, trend

# Momentum-Based Trading Strategy
def momentum_trading_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Momentum'] = momentum.RSIIndicator(stock_df['Close'], window).rsi()
    signals['signal'] = 'neutral'
    signals.loc[signals['Momentum'] > 70, 'signal'] = 'short'  # Overbought condition
    signals.loc[signals['Momentum'] < 30, 'signal'] = 'long'   # Oversold condition
    return signals[['signal']]
