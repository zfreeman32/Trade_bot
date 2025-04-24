
import pandas as pd
import numpy as np
from ta import momentum, trend

# Momentum Trading Strategy
def momentum_trading_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Momentum
    momentum_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = momentum_indicator.rsi()
    
    # Generate Buy/Sell signals based on RSI
    signals['momentum_signal'] = 'neutral'
    signals.loc[(signals['RSI'] < 30), 'momentum_signal'] = 'long'  # Buy signal
    signals.loc[(signals['RSI'] > 70), 'momentum_signal'] = 'short'  # Sell signal
    
    # Drop the RSI column to keep only signals
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals
