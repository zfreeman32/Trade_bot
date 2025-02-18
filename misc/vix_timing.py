import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

def vix_timing_signals(vix_df, sma_length=14, trend_length=14):
    signals = pd.DataFrame(index=vix_df.index)
    sma_indicator = SMAIndicator(vix_df['price'], window=sma_length)
    signals['sma'] = sma_indicator.sma_indicator()
    signals['price'] = vix_df['price']

    signals['buy_signal'] = 0
    signals['sell_signal'] = 0

    vix_above_sma = signals['price'] > signals['sma']
    vix_below_sma = signals['price'] < signals['sma']

    signals.loc[vix_below_sma & (vix_below_sma.shift() == False) & (vix_below_sma.rolling(window=trend_length).sum() == trend_length), 'buy_signal'] = 1
    signals.loc[vix_above_sma & (vix_above_sma.shift() == False) & (vix_above_sma.rolling(window=trend_length).sum() == trend_length), 'sell_signal'] = 1

    signals = signals.drop(['sma', 'price'], axis=1)

    return signals
