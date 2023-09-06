#%%
import numpy as np
import pandas as pd
from ta import trend

#%%
# ADX 
def adx_strength_direction(stock_df, window=14):
    # Create ADX indicator
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

    # Create a DataFrame to store the results
    signals = pd.DataFrame(index=stock_df.index)
    signals['ADX'] = adx.adx()
    signals['adx_Trend_Strength'] = None
    signals['adx_Direction'] = None

    # Determine trend strength and direction based on ADX values
    for i in range(window, len(signals)):
        if signals['ADX'][i] > 25:
            signals.loc[signals.index[i], 'adx_Trend_Strength'] = 'strong'
        elif signals['ADX'][i] < 20:
            signals.loc[signals.index[i], 'adx_Trend_Strength'] = 'weak'

        if adx.adx_pos()[i] > adx.adx_neg()[i]:
            signals.loc[signals.index[i], 'adx_Direction'] = 'bullish'
        elif adx.adx_pos()[i] < adx.adx_neg()[i]:
            signals.loc[signals.index[i], 'adx_Direction'] = 'bearish'

    return signals


#%%
# aroon
def aroon_strategy(stock_df, window=25):
    # Create Aroon Indicator
    aroon = trend.AroonIndicator(stock_df['Close'], window)

    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Aroon Up and Aroon Down values
    signals['Aroon_Up'] = aroon.aroon_up()
    signals['Aroon_Down'] = aroon.aroon_down()

    # Determine trend strength and save as 'aroon_Trend_Strength'
    signals['aroon_Trend_Strength'] = 'weak'
    signals.loc[(signals['Aroon_Up'] >= 70) | (signals['Aroon_Down'] >= 70), 'aroon_Trend_Strength'] = 'strong'

    # Determine bullish and bearish signals based on 'aroon_direction_signal'
    signals['aroon_direction_signal'] = 'bearish'
    signals.loc[signals['Aroon_Up'] > signals['Aroon_Down'], 'aroon_direction_signal'] = 'bullish'
    
    # Generate buy (1) and sell (-1) signals based on Aroon crossovers
    signals['Signal'] = 0
    for i in range(1, len(signals)):
        if signals['aroon_direction_signal'][i] == 'bullish' and signals['aroon_direction_signal'][i - 1] == 'bearish':
            signals['Signal'][i] = 'long'  # Buy signal (Aroon Up crosses above Aroon Down)
        elif signals['aroon_direction_signal'][i] == 'bearish' and signals['aroon_direction_signal'][i - 1] == 'bullish':
            signals['Signal'][i] = 'short'  # Sell signal (Aroon Down crosses above Aroon Up)

    return signals

#%%
csv_file = r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot\SPY.csv'
stock_df = pd.read_csv(csv_file)
ao_signals = pd.DataFrame(aroon_strategy(stock_df))
print(ao_signals)

# %%
# CCI
def cci_signals(stock_df, window=20, constant=0.015, overbought=100, oversold=-100):
    # Create CCI Indicator
    cci = trend.CCIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window, constant)

    # Create a DataFrame to store the trading signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['CCI'] = cci.cci()

    # Generate trading signals
    signals['cci_Signal'] = 0  # Initialize the signal column with zeros

    for i in range(window, len(signals)):
        cci_value = signals['CCI'][i]

        # Bullish market and buy signal
        if cci_value > oversold:
            signals.loc[signals.index[i], 'cci_direction'] = 'bullish'

        # Bearish market and sell signal
        elif cci_value < overbought:
            signals.loc[signals.index[i], 'cci_direction'] = 'bearish'

                # Bullish market and buy signal
        if cci_value > overbought and signals['CCI'][i - 1] <= overbought:
            signals.loc[signals.index[i], 'cci_Signal'] = 'buy'

        # Bearish market and sell signal
        elif cci_value < oversold and signals['CCI'][i - 1] >= oversold:
            signals.loc[signals.index[i], 'cci_Signal'] = 'sell'

    return signals

csv_file = r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot\SPY.csv'
stock_df = pd.read_csv(csv_file)
ao_signals = pd.DataFrame(cci_signals(stock_df))
print(ao_signals)
# %%
# dpo
def dpo_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['dpo_Signal'] = 0

    # Create the DPO indicator
    dpo = trend.DPOIndicator(stock_df['Close'], window)

    # Calculate the DPO values
    dpo_values = dpo.dpo()

    # Generate trading signals
    for i in range(window, len(signals)):
        if dpo_values[i] > 0:
            signals.loc[signals.index[i], 'dpo_direction_Signal'] = 'bullish'  # Bullish signal
        elif dpo_values[i] < 0:
            signals.loc[signals.index[i], 'dpo_direction_Signal'] = 'bearish'  # Bearish signal
        if dpo_values[i] > 0 and dpo_values[i - 1] <= 0:
            signals.loc[signals.index[i], 'dpo_Signal'] = 'buy'  # Buy signal (DPO crosses above zero)
        elif dpo_values[i] < 0 and dpo_values[i - 1] >= 0:
            signals.loc[signals.index[i], 'dpo_Signal'] = 'sell'
    return signals


# %%
# EMA
def ema_signals(stock_df, short_window=12, long_window=26):
    signals = pd.DataFrame(index=stock_df.index)
    signals['EMA_Signal'] = 0  # Initialize the signal column with zeros
    signals['EMA_Direction_Signal'] = 0
    # Calculate short-term EMA
    ema_short = stock_df['Close'].ewm(span=short_window, adjust=False).mean()

    # Calculate long-term EMA
    ema_long = stock_df['Close'].ewm(span=long_window, adjust=False).mean()

    # Generate EMA signals
    for i in range(1, len(stock_df)):
        if ema_short[i] > ema_long[i] and ema_short[i - 1] <= ema_long[i - 1]:
            signals.loc[stock_df.index[i], 'EMA_Signal'] = 'buy'  # Bullish (Buy) Signal
        elif ema_short[i] < ema_long[i] and ema_short[i - 1] >= ema_long[i - 1]:
            signals.loc[stock_df.index[i], 'EMA_Signal'] = 'sell'  # Bearish (Sell) Signal
        if ema_short[i] > ema_long[i]:
            signals.loc[stock_df.index[i], 'EMA_Direction_Signal'] = 'bullish'  # Bullish (Buy) Signal
        elif ema_short[i] < ema_long[i]:
            signals.loc[stock_df.index[i], 'EMA_Direction_Signal'] = 'bearish'  # Bearish (Sell) Signal

    return signals

#%%
# Ichi
def ichimoku_signals(stock_df, window1=9, window2=26):
    # Create the Ichimoku Indicator
    ichimoku = trend.IchimokuIndicator(stock_df['High'], stock_df['Low'], window1, window2)

    # Create a DataFrame to store the signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['ichi_signal'] = 0
    signals['ichi_direction'] = 0


    # Calculate Tenkan-sen and Kijun-sen values
    tenkan_sen = ichimoku.ichimoku_conversion_line()
    kijun_sen = ichimoku.ichimoku_base_line()
    senkou_span_a = ichimoku.ichimoku_a()
    senkou_span_b = ichimoku.ichimoku_b()
    cloud_color = ['green' if a > b else 'red' for a, b in zip(senkou_span_a, senkou_span_b)]

    # Generate signals based on crossovers
    for i in range(1, len(signals)):
        if tenkan_sen[i] > kijun_sen[i] and tenkan_sen[i - 1] <= kijun_sen[i - 1]:
            signals.loc[signals.index[i], 'ichi_signal'] = 'buy'
        elif tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i - 1] >= kijun_sen[i - 1]:
            signals.loc[signals.index[i], 'ichi_signal'] = 'sell'

        if tenkan_sen[i] > kijun_sen[i] :
            signals.loc[signals.index[i], 'ichi_direction'] = 'bullish'
        elif tenkan_sen[i] < kijun_sen[i] :
            signals.loc[signals.index[i], 'ichi_direction'] = 'bearish'

        if cloud_color[i] == 'green':
            signals.loc[signals.index[i], 'ichi_direction'] = 'bullish'
        else:
            signals.loc[signals.index[i], 'ichi_direction'] = 'bearish'
   

    return signals


# %%
# KST
def kst_signals(stock_df, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9):
    # Create KST Indicator
    kst = trend.KSTIndicator(stock_df['Close'], roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig)

    # Create a DataFrame to store the trading signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['kst_signal'] = 0 
    signals['kst_direction'] = 0 # Initialize signals with zeros

    # Calculate KST and its signal line
    kst_values = kst.kst()
    kst_signal_line = kst.kst_sig()

    # Generate bullish (1) and bearish (-1) signals based on KST crossovers
    for i in range(1, len(stock_df)):
        if kst_values[i] > kst_signal_line[i] and kst_values[i - 1] <= kst_signal_line[i - 1]:
            signals.loc[signals.index[i], 'kst_signal'] = 'buy'  # Bullish crossover
        elif kst_values[i] < kst_signal_line[i] and kst_values[i - 1] >= kst_signal_line[i - 1]:
            signals.loc[signals.index[i], 'kst_signal'] = 'sell'  # Bearish crossover

        if kst_values[i] > kst_signal_line[i]:
            signals.loc[signals.index[i], 'kst_direction'] = 'bullish'  # Bullish crossover
        elif kst_values[i] < kst_signal_line[i] :
            signals.loc[signals.index[i], 'kst_direction'] = 'bearish'  # Bearish crossover


    return signals


# %%
# MACD

def macd_signals(stock_df, window_slow=26, window_fast=12, window_sign=9):
    # Create MACD indicator
    macd = trend.MACD(stock_df['Close'], window_slow, window_fast, window_sign)

    # Create a DataFrame to store the trading signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['macd_signal'] = 0  # Initialize the signal column with zeros
    signals['macd_direction'] = 0

    # Calculate MACD values
    macd_line = macd.macd()
    signal_line = macd.macd_signal()

    # Generate trading signals
    for i in range(1, len(signals)):
        if macd_line[i] > signal_line[i] and macd_line[i - 1] <= signal_line[i - 1]:
            signals.loc[signals.index[i], 'macd_signal'] = 'buy'  # Bullish crossover (Buy signal)
        elif macd_line[i] < signal_line[i] and macd_line[i - 1] >= signal_line[i - 1]:
            signals.loc[signals.index[i], 'macd_signal'] = 'sell'  # Bearish crossover (Sell signal)

        if macd_line[i] > signal_line[i] :
            signals.loc[signals.index[i], 'macd_direction'] = 'bullish'
        elif macd_line[i] < signal_line[i] :
            signals.loc[signals.index[i], 'macd_direction'] = 'bearish'

    return signals



# %%
# Mass Index

# NEEDS WORK


def mass_index_signals(stock_data, fast_window=9, slow_window=25):
    # Create a DataFrame to store the trading signals
    signals = pd.DataFrame(index=stock_data.index)
    signals['Signal'] = 0

    # Calculate Mass Index
    mass_index = trend.MassIndex(stock_data['High'], stock_data['Low'], fast_window, slow_window)
    
    # Calculate Short and Long EMAs
    short_ema = trend.EMAIndicator(stock_data['Close'], window=fast_window)
    long_ema = trend.EMAIndicator(stock_data['Close'], window=slow_window)

    # Calculate the Mass Index and its reversal thresholds
    mass_index_values = mass_index.mass_index()
    reversal_bulge_threshold = 27
    reversal_bulge_exit_threshold = 26.50

    # Generate trading signals
    in_downtrend = short_ema.ema_indicator() > long_ema.ema_indicator()

    for i in range(len(signals)):
        if in_downtrend[i] is True and mass_index_values[i] > reversal_bulge_threshold and mass_index_values[i - 1] <= reversal_bulge_exit_threshold:
            signals.loc[signals.index[i], 'Signal'] = 'Buy'
        elif in_downtrend[i] is False and mass_index_values[i] > reversal_bulge_threshold and mass_index_values[i - 1] <= reversal_bulge_exit_threshold:
            signals.loc[signals.index[i], 'Signal'] = 'Sell'

    return signals


# %%
# PSAR
def psar_signals(stock_df, step=0.02, max_step=0.2):
    signals = pd.DataFrame(index=stock_df.index)
    signals['psar_direction'] = ''
    signals['psar_signal'] = ''

    # Calculate Parabolic SAR (PSAR)
    psar = trend.PSARIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], step, max_step)
    print(psar)
    #%%
    # PSAR
    # NEEDS WORK
    for i in range(1, len(signals)):
        if stock_df['Close'][i] > psar[i] and stock_df['Close'][i - 1] <= psar[i - 1]:
            signals.loc[signals.index[i], 'psar_signal'] = 'buy'  # Bullish crossover (Buy signal)
        elif stock_df['Close'][i] < psar[i] and stock_df['Close'][i - 1] >= psar[i - 1]:
            signals.loc[signals.index[i], 'psar_signal'] = 'sell'  # Bearish crossover (Sell signal)

        if stock_df['Close'][i] > psar[i] :
            signals.loc[signals.index[i], 'psar_direction'] = 'bullish'
        elif stock_df['Close'][i] < psar[i] :
            signals.loc[signals.index[i], 'psar_direction'] = 'bearish'

    return signals


# %%
# Golden Cross SMA

def golden_ma_signals(stock_df, short_period=50, long_period=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['ma_direction'] = ''
    signals['ma_signal'] = ''

    # Calculate short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period)
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_sma.sma_indicator() > long_sma.sma_indicator(), 'ma_direction'] = 'bullish'
    signals.loc[short_sma.sma_indicator() <= long_sma.sma_indicator(), 'ma_direction'] = 'bearish'

    # Generate buy and sell signals
    signals.loc[(short_sma.sma_indicator() > long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) <= long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'buy'
    signals.loc[(short_sma.sma_indicator() <= long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) > long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'sell'

    return signals


# %%
# 13-26 MA STrategy
def short_ma_signals(stock_df, short_period=13, long_period=26):
    signals = pd.DataFrame(index=stock_df.index)
    signals['ma_direction'] = ''
    signals['ma_signal'] = ''

    # Calculate short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period)
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_sma.sma_indicator() > long_sma.sma_indicator(), 'ma_direction'] = 'bullish'
    signals.loc[short_sma.sma_indicator() <= long_sma.sma_indicator(), 'ma_direction'] = 'bearish'

    # Generate buy and sell signals
    signals.loc[(short_sma.sma_indicator() > long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) <= long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'buy'
    signals.loc[(short_sma.sma_indicator() <= long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) > long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'sell'

    return signals

def strategy_5_8_13(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['5_8_13_signal'] = ''

    # Calculate SMAs with periods 5, 8, and 13
    sma5 = trend.SMAIndicator(stock_df['Close'], 5)
    sma8 = trend.SMAIndicator(stock_df['Close'], 8)
    sma13 = trend.SMAIndicator(stock_df['Close'], 13)

    # Determine the market direction (Bullish or Bearish)
    signals.loc[(sma5.sma_indicator() > sma8.sma_indicator()) &
                (sma8.sma_indicator() > sma13.sma_indicator()), '5_8_13_signal'] = 'bearish'

    signals.loc[(sma5.sma_indicator() < sma8.sma_indicator()) &
                (sma8.sma_indicator() < sma13.sma_indicator()), '5_8_13_signal'] = 'bullish'

    return signals


# %%
# STC
def stc_signals(stock_df, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stc_signal'] = 0
    signals['stc_direction'] = ''

    # Calculate STC (Stochastic RSI)
    stc = trend.STCIndicator(stock_df['Close'], window_slow, window_fast, cycle, smooth1, smooth2)

    # Determine overbought/oversold conditions
    overbought_condition = stc > 75
    oversold_condition = stc < 25

    # Determine bullish/bearish trend
    bullish_condition = stc > 50
    bearish_condition = stc < 50

    # Generate signals and directions
    signals.loc[overbought_condition, 'stc_signal'] = 'overbought'
    signals.loc[oversold_condition, 'stc_signal'] = 'oversold'

    signals.loc[bullish_condition, 'stc_direction'] = 'Bullish'
    signals.loc[bearish_condition, 'stc_direction'] = 'Bearish'


    return signals



# %%
# Vortex

# NEEDS WORK

def vortex_signals(stock_df, window=14, threshold=1.0):
    signals = pd.DataFrame(index=stock_df.index)
    signals['vortex_signal'] = 0
    signals['vortex_direction_signal'] = 0

    # Calculate the Vortex Indicator
    vortex = trend.VortexIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

    # Create signals based on the crossover of the positive and negative vortex indicators
    signals['Positive'] = vortex.vortex_indicator_pos
    signals['Negative'] = vortex.vortex_indicator_neg

    for i in range(1, len(signals)):
        if signals['Positive'][i] > signals['Negative'][i] and signals['Positive'][i - 1] <= signals['Negative'][i - 1]:
            signals.loc[signals.index[i], 'vortex_signal'] = 'buy'  # Buy signal
        elif signals['Positive'][i] < signals['Negative'][i] and signals['Positive'][i - 1] >= signals['Negative'][i - 1]:
            signals.loc[signals.index[i], 'vortex_signal'] = 'sell'  # Sell signal

        if signals['Positive'][i] > signals['Negative'][i] :
            signals.loc[signals.index[i], 'vortex_direction_signal'] = 'bullish'  # Buy signal
        elif signals['Positive'][i] < signals['Negative'][i]:
            signals.loc[signals.index[i], 'vortex_direction_signal'] = 'bearish'

    return signals
csv_file = r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot\SPY.csv'
stock_df = pd.read_csv(csv_file)
ao_signals = pd.DataFrame(vortex_signals(stock_df))
print(ao_signals)
# %%
# Weighted Moving Average
# Golden Cross WMA

def golden_wma_signals(stock_df, short_period=50, long_period=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['wma_direction'] = ''
    signals['wma_signal'] = ''

    # Calculate short and long SMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period)
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_wma.wma() > long_wma.wma, 'wma_direction'] = 'bullish'
    signals.loc[short_wma.wma() <= long_wma.wma(), 'mwa_direction'] = 'bearish'

    # Generate buy and sell signals
    signals.loc[(short_wma.wma() > long_wma.wma()) &
                (short_wma.wma().shift(1) <= long_wma.wma().shift(1)), 'wma_signal'] = 'buy'
    signals.loc[(short_wma.wma() <= long_wma.wma()) &
                (short_wma.wma().shift(1) > long_wma.wma().shift(1)), 'wma_signal'] = 'sell'

    return signals

# 13-26 MA STrategy
def short_wma_signals(stock_df, short_period=13, long_period=26):
    signals = pd.DataFrame(index=stock_df.index)
    signals['wma_direction'] = ''
    signals['wma_signal'] = ''

    # Calculate short and long SMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period)
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_wma.wma() > long_wma.wma, 'wma_direction'] = 'bullish'
    signals.loc[short_wma.wma() <= long_wma.wma(), 'wma_direction'] = 'bearish'

    # Generate buy and sell signals
    signals.loc[(short_wma.wma() > long_wma.wma()) &
                (short_wma.wma().shift(1) <= long_wma.wma().shift(1)), 'wma_signal'] = 'buy'
    signals.loc[(short_wma.wma() <= long_wma.wma()) &
                (short_wma.wma().shift(1) > long_wma.wma().shift(1)), 'wma_signal'] = 'sell'

    return signals

def strategy_w5_8_13(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['w5_8_13_signal'] = ''

    # Calculate SMAs with periods 5, 8, and 13
    wma5 = trend.WMAIndicator(stock_df['Close'], 5)
    wma8 = trend.WMAIndicator(stock_df['Close'], 8)
    wma13 = trend.WMAIndicator(stock_df['Close'], 13)

    # Determine the market direction (Bullish or Bearish)
    signals.loc[(wma5.wma() > wma8.wma()) &
                (wma8.wma() > wma13.wma()), 'w5_8_13_signal'] = 'bearish'

    signals.loc[(wma5.wma() < wma8.wma()) &
                (wma8.wma() < wma13.wma()), 'w5_8_13_signal'] = 'bullish'

    return signals
