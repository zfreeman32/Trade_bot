import pandas as pd
from ta import trend

def adx_breakouts_signals(stock_df, highest_length=15, adx_length=14, adx_level=40, offset=0.5):
    signals = pd.DataFrame(index=stock_df.index)
    highest = stock_df['High'].rolling(window=highest_length).max()
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_length).adx()

    signals['adx'] = adx
    signals['highest'] = highest
    signals['adx_breakout_signal'] = 'neutral'
    signals.loc[(signals['adx'] > adx_level) & (stock_df['Close'] > (signals['highest'] + offset)), 'adx_breakout_signal'] = 'long'
    signals.drop(['adx', 'highest'], axis=1, inplace=True)
    return signals

