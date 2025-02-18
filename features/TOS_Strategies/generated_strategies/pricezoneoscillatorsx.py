#%%
import pandas as pd
from ta import trend
def calculate_PZO(df, length=14):
    EMA_high = df['High'].ewm(span=length, adjust=False).mean()
    EMA_low = df['Low'].ewm(span=length, adjust=False).mean()
    EMA_close = df['Close'].ewm(span=length, adjust=False).mean()
    donchian_channel = EMA_high - EMA_low
    y = (EMA_close - ((EMA_high + EMA_low)/2)) / donchian_channel
    PZO = y*100
    return PZO

def PriceZoneOscillatorSX(df, length=14, ema_length=60):
    signals = pd.DataFrame(index=df.index)
    adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'], length).adx()
    ema = df['Close'].ewm(span=ema_length, adjust=False).mean()
    PZO = calculate_PZO(df, length)

    signals['adx'] = adx
    signals['ema'] = ema
    signals['pzo'] = PZO

    # Initialize short_exit to zero
    signals['pzo_sx_signal'] = 0

    # Calculate short_exit conditions based on PZO, ADX and EMA values
    for i in range(2, signals.shape[0]):
        if (signals['adx'].iloc[i] > 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                (signals['pzo'].iloc[i] > 0 and df['Close'].iloc[i] > signals['ema'].iloc[i])):
                signals.loc[signals.index[i], 'pzo_sx_signal'] = 1
        elif (signals['adx'].iloc[i] < 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                ((signals['pzo'].iloc[i] > 0 or signals['pzo'].iloc[i-1] > -40) and df['Close'].iloc[i] > signals['ema'].iloc[i]) or
                (signals['pzo'].iloc[i] > 15 and signals['pzo'].iloc[i-1] <= -5 and signals['pzo'].iloc[i-2] > -40)):
                signals.loc[signals.index[i], 'pzo_sx_signal'] = 1
        
    signals.drop(['pzo', 'ema', 'adx'], axis=1, inplace=True)

    return signals
