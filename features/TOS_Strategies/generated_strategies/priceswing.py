
from ta import momentum, volatility
import numpy as np
import pandas as pd

def price_swing_signals(stock_df, swing_type="RSI", length=20, exit_length=20, deviations=2, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=stock_df['Close'], window=length, window_dev=deviations)
    stock_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    stock_df['bb_bbh'] = indicator_bb.bollinger_hband()
    stock_df['bb_bbl'] = indicator_bb.bollinger_lband()

    if swing_type == "bollinger":
        # Use Bollinger Bands crossover swing type
        signals['price_swing_signals'] = np.where(stock_df['Close'] > stock_df['bb_bbh'], 'down', 'neutral')
        signals['price_swing_signals'] = np.where(stock_df['Close'] < stock_df['bb_bbl'], 'up', signals['price_swing_signals'])

    elif swing_type == "RSI":
        # Use RSI crossover swing type
        rsi = momentum.RSIIndicator(close=stock_df['Close'], window=length)
        signals['rsi'] = rsi.rsi()
        signals['price_swing_signals'] = np.where(signals['rsi'] > overbought, 'down', 'neutral')
        signals['price_swing_signals'] = np.where(signals['rsi'] < oversold, 'up', signals['price_swing_signals'])
        signals.drop(['rsi'], axis=1, inplace=True)

    elif swing_type == "RSI_HighLow":
        # Use RSI + Higher Low/Lower High swing type
        # Insert your RSI + Higher Low/Lower High swing type strategy here. 
        pass

    # Add simulated exit orders after exit_length
    signals['exit'] = signals['price_swing_signals'].shift(exit_length)
        
    signals.drop(['exit'], axis=1, inplace=True)

    return signals

