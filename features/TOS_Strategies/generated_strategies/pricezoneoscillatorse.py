import pandas as pd
import numpy as np
from ta import momentum, trend, volume

def pzo_signals(stock_df, length=14, ema_length=60):
    signals = pd.DataFrame(index=stock_df.index)

    # compute indicators
    adx_ind = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], length)
    ema_ind = trend.EMAIndicator(stock_df['Close'], ema_length)

    # Calculate PriceZoneOscillator (the formula mentioned in the reference)
    # The calculation form of PZO is not mentioned explicitly in your description - variance in this may affect the output.
    vpt_ind = volume.VolumePriceTrendIndicator(stock_df['Close'], stock_df['Volume'])
    pzo = vpt_ind.volume_price_trend()
    signals['PZO'] = pzo
    signals['EMA'] = ema_ind.ema_indicator()
    signals['ADX'] = adx_ind.adx()

    # Define conditions
    adx_trending = signals['ADX'] > 18
    adx_not_trending = signals['ADX'] < 18
    price_below_ema = stock_df['Close'] < signals['EMA']
    cross_above_40 = (signals['PZO'] > 40) & (signals['PZO'].shift(1) < 40)
    fall_below_minus5 = (signals['PZO'] < -5) & (signals['PZO'].shift(1) > -5)
    cross_above_zero = (signals['PZO'] > 0) & (signals['PZO'].shift(1) < 0)

    # Define signals based on conditions
    signals['pzo_se_signal'] = 'neutral'
    signals.loc[adx_trending & price_below_ema & (cross_above_40 | (cross_above_zero & fall_below_minus5)), 'pzo_se_signal'] = 'short'
    signals.loc[adx_not_trending & (cross_above_40 | fall_below_minus5), 'pzo_se_signal'] = 'short'

    # Drop unnecessary columns to return only signals
    signals.drop(['PZO', 'EMA', 'ADX'], axis=1, inplace=True)

    return signals

