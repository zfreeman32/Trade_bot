
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume

# EURCAD Forex Trading Strategy
def eurcad_trading_signals(forex_df, short_window=10, long_window=30, atr_period=14):
    signals = pd.DataFrame(index=forex_df.index)
    
    # Calculate moving averages
    signals['short_mavg'] = forex_df['Close'].rolling(window=short_window).mean()
    signals['long_mavg'] = forex_df['Close'].rolling(window=long_window).mean()
    
    # Calculate Average True Range (ATR)
    signals['high_low'] = forex_df['High'] - forex_df['Low']
    signals['high_close'] = np.abs(forex_df['High'] - forex_df['Close'].shift(1))
    signals['low_close'] = np.abs(forex_df['Low'] - forex_df['Close'].shift(1))
    signals['atr'] = signals[['high_low', 'high_close', 'low_close']].max(axis=1).rolling(window=atr_period).mean()

    signals['signal'] = 'neutral'
    # Generate signals based on moving averages crossover
    signals.loc[(signals['short_mavg'] > signals['long_mavg']), 'signal'] = 'long'
    signals.loc[(signals['short_mavg'] < signals['long_mavg']), 'signal'] = 'short'
    
    # Filter signals based on ATR volatility
    signals['signal'] = np.where(signals['atr'] > signals['atr'].mean(), signals['signal'], 'neutral')

    return signals['signal']
