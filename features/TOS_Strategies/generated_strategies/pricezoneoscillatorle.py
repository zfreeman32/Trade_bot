import pandas as pd
import numpy as np
from ta import trend, volatility

# Price Zone Oscillator (PZO) Strategy
def pzo_signals(stock_df, length=14, ema_length=60):
    signals = pd.DataFrame(index=stock_df.index)
    pzo = ((stock_df['Close'] - stock_df['Close'].rolling(window=length).mean()) / stock_df['Close'].rolling(window=length).std())*100
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=length).adx()
    ema = stock_df['Close'].ewm(span=ema_length).mean()

    signals['PZO'] = pzo
    signals['ADX'] = adx
    signals['EMA'] = ema
    signals['pzo_le_signals'] = 'neutral'

    # ADX > 18, price > EMA, and PZO cross "-40" level or surpass "+15" level from below
    signals.loc[(signals['ADX'] > 18) & (stock_df['Close'] > signals['EMA']) & (
        (signals['PZO'].shift(1) < -40) & (signals['PZO'] > -40) |
        ((signals['PZO'].shift(1) < 0) & (signals['PZO'] > 0) & (signals['PZO'] > 15))), 'pzo_le_signals'] = 'long'
    
    # ADX < 18, and PZO cross "-40" or "+15" level from below
    signals.loc[(signals['ADX'] <= 18) & (
        (signals['PZO'].shift(1) < -40) & (signals['PZO'] > -40) |
        (signals['PZO'].shift(1) < 15) & (signals['PZO'] > 15)), 'pzo_le_signals'] = 'long'

    signals.drop(['PZO','ADX','EMA'], axis=1, inplace=True)
    return signals
