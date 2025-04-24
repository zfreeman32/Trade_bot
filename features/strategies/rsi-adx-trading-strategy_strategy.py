
import pandas as pd
import numpy as np
from ta import momentum, trend

# RSI & ADX Strategy
def rsi_adx_signals(stock_df, rsi_window=14, adx_window=14, rsi_overbought=70, rsi_oversold=30):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window=rsi_window)
    signals['RSI'] = rsi.rsi()

    # Calculate ADX
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_window)
    signals['ADX'] = adx.adx()

    # Generate trading signals
    signals['signal'] = 'neutral'
    
    # Long signal: when RSI < 30 and ADX > 20
    signals.loc[(signals['RSI'] < rsi_oversold) & (signals['ADX'] > 20), 'signal'] = 'long'
    
    # Short signal: when RSI > 70 and ADX > 20
    signals.loc[(signals['RSI'] > rsi_overbought) & (signals['ADX'] > 20), 'signal'] = 'short'

    # Drop RSI and ADX columns, as we only need the signals
    signals.drop(['RSI', 'ADX'], axis=1, inplace=True)

    return signals
