#%%
from ta import momentum, trend, volatility, volume
import numpy as np
import pandas as pd

#%%
# PPO
def ppo_signals(stock_data, fast_window=12, slow_window=26, signal_window=9):
    signals = pd.DataFrame(index=stock_data.index)
    signals['PPO_signal'] = 0

    ppo = momentum.PercentagePriceOscillator(stock_data['Close'], fast_window, slow_window, signal_window)

    ppo_values = ppo.ppo()
    ppo_signal = ppo.ppo_signal()

    # Generate long (1) and short (-1) signals based on PPO and its signal line
    for i in range(1, len(stock_data)):
        if ppo_values[i] > ppo_signal[i] and ppo_values[i - 1] <= ppo_signal[i - 1]:
            signals.loc[stock_data.index[i], 'PPO_signal'] = 'long'  # long signal (PPO crosses above signal line)
        elif ppo_values[i] < ppo_signal[i] and ppo_values[i - 1] >= ppo_signal[i - 1]:
            signals.loc[stock_data.index[i], 'PPO_signal'] = 'short'  # short signal (PPO crosses below signal line)

    return signals

#%%
# Awesome Oscilator 0 cross
def Awesome_Oscillator_signals(stock_df):
    # Define long and short signals
    signals = pd.DataFrame(index=stock_df.index)
    ao_indicator = momentum.AwesomeOscillatorIndicator(high=stock_df['High'], low=stock_df['Low'])
    signals['Ao_signal'] = 0
    stock_df['momentum_ao'] = ao_indicator.awesome_oscillator()

    for i in range(1, len(stock_df)):
        if (
            stock_df['momentum_ao'].iloc[i-1] < 0 and
            stock_df['momentum_ao'].iloc[i] >= 0
        ):
            signals['Ao_signal'].iloc[i] = 'long' # long signal
        elif (
            stock_df['momentum_ao'].iloc[i-1] >= 0 and
            stock_df['momentum_ao'].iloc[i] < 0
        ):
            signals['Ao_signal'].iloc[i] = 'short'  # short signal
    
    return signals

#%%
# KAMA Cross
def kama_cross_signals(stock_df, fast_period=10, slow_period=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['kama_cross_signal'] = 0
    signals['kama_signal'] = 0

    # Calculate Fast KAMA
    fast_kama = momentum.kama(stock_df['Close'], window=fast_period)

    # Calculate Slow KAMA
    slow_kama = momentum.kama(stock_df['Close'], window=slow_period)
    for i in range(1, len(stock_df)):
        if fast_kama[i] > slow_kama[i] and fast_kama[i - 1] <= slow_kama[i - 1] and stock_df['Close'][i] > fast_kama[i]:
            signals['kama_cross_signal'].iloc[i] = 'long'  # long signal (fast KAMA above slow KAMA and price above fast KAMA)
        elif fast_kama[i] < slow_kama[i] and fast_kama[i - 1] >= slow_kama[i - 1] and stock_df['Close'][i] < fast_kama[i]:
            signals['kama_cross_signal'].iloc[i] = 'short'  # short signal (fast KAMA below slow KAMA and price below fast KAMA)
        elif stock_df['Close'][i] > fast_kama[i] and stock_df['Close'][i - 1] <= fast_kama[i - 1]:
            stock_df.loc[i, 'kama_signal'] = 'long' # long signal (price crosses above KAMA)
        elif stock_df['Close'][i] < fast_kama[i] and stock_df['Close'][i - 1] >= fast_kama[i - 1]:
            stock_df.loc[i, 'kama_signal'] = 'short'

    return signals

#%%
# Stoch
def stoch_signals(stock_df, fast_period=10, slow_period=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stoch_signal'] = 0

    # Calculate Stochastic Oscillator
    stoch = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=14, smooth_window=3)

    signals['%K'] = stoch.stoch()
    signals['%D'] = stoch.stoch_signal()

    # Generate long (1) and short (-1) signals based on Stochastic Oscillator
    for i in range(1, len(signals)):
        if signals['%K'][i] > signals['%D'][i] and signals['%K'][i - 1] <= signals['%D'][i - 1]:
            signals.loc[i, 'stoch_signal'] = 'long'  # long signal (%K crosses above %D)
        elif signals['%K'][i] < signals['%D'][i] and signals['%K'][i - 1] >= signals['%D'][i - 1]:
            signals.loc[i, 'stoch_signal'] = 'short'  # short signal (%K crosses below %D)

    signals.drop(['%K', '%D'], axis=1, inplace=True)

    return signals

#%%
# TSI 
def tsi_signals(stock_df, window_slow=25, window_fast=13):
    signals = pd.DataFrame(index=stock_df.index)
    signals['tsi_signal'] = 0

    # Calculate True Strength Index (TSI)
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)

    signals['TSI'] = tsi.tsi()

    # Generate long (1) and short (-1) signals based on TSI
    for i in range(1, len(signals)):
        if signals['TSI'][i] > 0 and signals['TSI'][i - 1] <= 0:
            signals.loc[i, 'tsi_signal'] = 'long'  # long signal (TSI crosses above 0)
        elif signals['TSI'][i] < 0 and signals['TSI'][i - 1] >= 0:
            signals.loc[i, 'tsi_signal'] = 'short'  # short signal (TSI crosses below 0)

    signals.drop(['TSI'], axis=1, inplace=True)

    return signals

# %%
# Williams R overbough/oversold
def williams_signals(stock_df, lbp=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['williams_signal'] = 0

    # Calculate Williams %R
    williams_r = momentum.WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], lbp)

    signals['WilliamsR'] = williams_r.williams_r()

    # Generate overbought (1) and oversold (-1) signals based on Williams %R
    for i in range(len(signals)):
        if signals['WilliamsR'][i] <= -80:
            signals.loc[i, 'williams_signal'] = 'overbought'  # Overbought signal (Williams %R crosses below or equal to -80)
        elif signals['WilliamsR'][i] >= -20:
            signals.loc[i, 'williams_signal'] = 'oversold'  # Oversold signal (Williams %R crosses above or equal to -20)

    signals.drop(['WilliamsR'], axis=1, inplace=True)

    return signals

# %%
# ROC overbough/oversold
def roc_signals(stock_df, window=12):
    signals = pd.DataFrame(index=stock_df.index)
    signals['roc_signal'] = 0

    # Calculate Rate of Change (ROC)
    roc = momentum.ROCIndicator(stock_df['Close'], window)

    signals['ROC'] = roc.roc()

    # Generate overbought (1) and oversold (-1) signals based on ROC
    for i in range(1, len(signals)):
        if signals['ROC'][i] >= 10:
            signals.loc[i, 'roc_signal'] = 'overbought'  # Overbought signal (ROC crosses above or equal to 10)
        elif signals['ROC'][i] <= -10:
            signals.loc[i, 'roc_signal'] = 'oversold'  # Oversold signal (ROC crosses below or equal to -10)

    signals.drop(['ROC'], axis=1, inplace=True)

    return signals

#%%
# RSI overbough/oversold
def rsi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['rsi_signal'] = 0

    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window)

    signals['RSI'] = rsi.rsi()

    # Generate overbought (1) and oversold (-1) signals based on RSI
    for i in range(1, len(signals)):
        if signals['RSI'][i] >= 70:
            signals.loc[i, 'rsi_signal'] = 'overbought'  # Overbought signal (RSI crosses above or equal to 70)
        elif signals['RSI'][i] <= 30:
            signals.loc[i, 'rsi_signal'] = 'oversold'  # Oversold signal (RSI crosses below or equal to 30)
    
    signals.drop(['RSI'], axis=1, inplace=True)

    return signals

# %%
# Stoch RSI
def stochrsi_signals(stock_df, window=14, smooth1=3, smooth2=3):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stochrsi_signal'] = 0

    # Calculate StochRSI
    stoch_rsi = momentum.StochRSIIndicator(stock_df['Close'], window, smooth1, smooth2)

    signals['StochRSI'] = stoch_rsi.stochrsi()

    # Generate overbought (1) and oversold (-1) signals based on StochRSI
    for i in range(1, len(signals)):
        if signals['StochRSI'][i] >= 0.8:
            signals.loc[i, 'stochrsi_signal'] = 'overbought'  # Overbought signal (StochRSI crosses above or equal to 0.8)
        elif signals['StochRSI'][i] <= 0.2:
            signals.loc[i, 'stochrsi_signal'] = 'oversold'  # Oversold signal (StochRSI crosses below or equal to 0.2)
    
    signals.drop(['StochRSI'], axis=1, inplace=True)

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
    
    # Generate long (1) and short (-1) signals based on Aroon crossovers
    signals['aroon_signal'] = 0
    for i in range(1, len(signals)):
        if signals['aroon_direction_signal'][i] == 'bullish' and signals['aroon_direction_signal'][i - 1] == 'bearish':
            signals.loc[signals.index[i], 'aroon_signal'] = 'long'  # long signal (Aroon Up crosses above Aroon Down)
        elif signals['aroon_direction_signal'][i] == 'bearish' and signals['aroon_direction_signal'][i - 1] == 'bullish':
            signals.loc[signals.index[i], 'aroon_signal'] = 'short'  # short signal (Aroon Down crosses above Aroon Up)

    signals.drop(['Aroon_Up', 'Aroon_Down'], axis=1, inplace=True)

    return signals

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

        # Bullish market and long signal
        if cci_value > oversold:
            signals.loc[signals.index[i], 'cci_direction'] = 'bullish'

        # Bearish market and short signal
        elif cci_value < overbought:
            signals.loc[signals.index[i], 'cci_direction'] = 'bearish'

                # Bullish market and long signal
        if cci_value > overbought and signals['CCI'][i - 1] <= overbought:
            signals.loc[signals.index[i], 'cci_Signal'] = 'long'

        # Bearish market and short signal
        elif cci_value < oversold and signals['CCI'][i - 1] >= oversold:
            signals.loc[signals.index[i], 'cci_Signal'] = 'short'
    signals.drop(['CCI'], axis=1, inplace=True)

    return signals

# %%
# dpo
def dpo_signals(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['dpo_Signal'] = 0
    signals['dpo_direction_Signal'] = 0

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
            signals.loc[signals.index[i], 'dpo_Signal'] = 'long'  # long signal (DPO crosses above zero)
        elif dpo_values[i] < 0 and dpo_values[i - 1] >= 0:
            signals.loc[signals.index[i], 'dpo_Signal'] = 'short'
    
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
            signals.loc[stock_df.index[i], 'EMA_Signal'] = 'long'  # Bullish (long) Signal
        elif ema_short[i] < ema_long[i] and ema_short[i - 1] >= ema_long[i - 1]:
            signals.loc[stock_df.index[i], 'EMA_Signal'] = 'short'  # Bearish (short) Signal
        if ema_short[i] > ema_long[i]:
            signals.loc[stock_df.index[i], 'EMA_Direction_Signal'] = 'bullish'  # Bullish (long) Signal
        elif ema_short[i] < ema_long[i]:
            signals.loc[stock_df.index[i], 'EMA_Direction_Signal'] = 'bearish'  # Bearish (short) Signal

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
            signals.loc[signals.index[i], 'ichi_signal'] = 'long'
        elif tenkan_sen[i] < kijun_sen[i] and tenkan_sen[i - 1] >= kijun_sen[i - 1]:
            signals.loc[signals.index[i], 'ichi_signal'] = 'short'

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
            signals.loc[signals.index[i], 'kst_signal'] = 'long'  # Bullish crossover
        elif kst_values[i] < kst_signal_line[i] and kst_values[i - 1] >= kst_signal_line[i - 1]:
            signals.loc[signals.index[i], 'kst_signal'] = 'short'  # Bearish crossover

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
            signals.loc[signals.index[i], 'macd_signal'] = 'long'  # Bullish crossover (long signal)
        elif macd_line[i] < signal_line[i] and macd_line[i - 1] >= signal_line[i - 1]:
            signals.loc[signals.index[i], 'macd_signal'] = 'short'  # Bearish crossover (short signal)

        if macd_line[i] > signal_line[i] :
            signals.loc[signals.index[i], 'macd_direction'] = 'bullish'
        elif macd_line[i] < signal_line[i] :
            signals.loc[signals.index[i], 'macd_direction'] = 'bearish'

    return signals

# %%
# Golden Cross SMA
def golden_ma_signals(stock_df, short_period=50, long_period=200):
    signals = pd.DataFrame(index=stock_df.index)
    signals['ma_direction'] = 0
    signals['ma_signal'] = 0

    # Calculate short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period)
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_sma.sma_indicator() > long_sma.sma_indicator(), 'ma_direction'] = 'bullish'
    signals.loc[short_sma.sma_indicator() <= long_sma.sma_indicator(), 'ma_direction'] = 'bearish'

    # Generate long and short signals
    signals.loc[(short_sma.sma_indicator() > long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) <= long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'long'
    signals.loc[(short_sma.sma_indicator() <= long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) > long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'short'

    return signals

# %%
# 13-26 MA STrategy
def short_ma_signals(stock_df, short_period=13, long_period=26):
    signals = pd.DataFrame(index=stock_df.index)
    signals['ma_direction'] = 0
    signals['ma_signal'] = 0

    # Calculate short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period)
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period)

    # Determine market direction (Bullish or Bearish)
    signals.loc[short_sma.sma_indicator() > long_sma.sma_indicator(), 'ma_direction'] = 'bullish'
    signals.loc[short_sma.sma_indicator() <= long_sma.sma_indicator(), 'ma_direction'] = 'bearish'

    # Generate long and short signals
    signals.loc[(short_sma.sma_indicator() > long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) <= long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'long'
    signals.loc[(short_sma.sma_indicator() <= long_sma.sma_indicator()) &
                (short_sma.sma_indicator().shift(1) > long_sma.sma_indicator().shift(1)), 'ma_signal'] = 'short'

    return signals

#%%
def strategy_5_8_13(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['5_8_13_signal'] = 0

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

#%%
def strategy_w5_8_13(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['w5_8_13_signal'] = 0

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

#%%
def atr_signals(stock_df, atr_window=14, ema_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['trend_strength'] = 0

    # Calculate Average True Range (ATR)
    atr = volatility.AverageTrueRange(stock_df['High'], stock_df['Low'], stock_df['Close'], atr_window).average_true_range()

    # Calculate 20-period Exponential Moving Average (EMA)
    ema = trend.EMAIndicator(atr, ema_window).ema_indicator()

    # Determine trend strength
    for i in range(ema_window, len(signals)):
        current_atr = atr[i]
        current_ema = ema[i]

        if current_atr > current_ema:
            signals.loc[signals.index[i], 'trend_strength'] = 'strong'
        else:
            signals.loc[signals.index[i], 'trend_strength'] = 'weak'

    return signals

#%%
# Define a function for the trading strategy
def keltner_channel_strategy(stock_df, window=20, window_atr=10, multiplier=2):
    # Create Keltner Channel indicator
    keltner_channel = volatility.KeltnerChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window, window_atr, multiplier)

    # Create a DataFrame to store the trading signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0

    # Calculate Keltner Channel values
    keltner_channel_upper = keltner_channel.keltner_channel_hband()
    keltner_channel_lower = keltner_channel.keltner_channel_lband()

    # Generate trading signals
    for i in range(window, len(signals)):
        if stock_df['Close'][i] > keltner_channel_upper[i]:
            signals.loc[signals.index[i], 'Signal'] = 'bearish'  # Bearish trend (Short signal)
        elif stock_df['Close'][i] < keltner_channel_lower[i]:
            signals.loc[signals.index[i], 'Signal'] = 'bullish'  # Bullish trend (Long signal)

    return signals

#%%
# Chaikin Money Flow
def cmf_signals(stock_df, window=20, threshold=0.1):
    signals = pd.DataFrame(index=stock_df.index)
    signals['cmf_signal'] = 0

    # Calculate Chaikin Money Flow (CMF)
    cmf = volume.ChaikinMoneyFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)

    signals['CMF'] = cmf.chaikin_money_flow()

    # Generate long (1) and short (-1) signals based on CMF
    for i in range(1, len(signals)):
        if signals['CMF'][i] > threshold:  # You can adjust this threshold as needed
            signals.loc[i, 'cmf_signal'] = 'bearish'  # long signal (CMF is positive)
        elif signals['CMF'][i] < -threshold:  # You can adjust this threshold as needed
            signals.loc[i, 'cmf_signal'] = 'bullish'  # short signal (CMF is negative)

    signals.drop(['CMF'], axis=1, inplace=True)

    return signals

#%%
# MFI
def mfi_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['mfi_signal'] = 0

    # Calculate Money Flow Index (MFI)
    mfi = volume.MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)

    signals['MFI'] = mfi.money_flow_index()

    # Generate long (1) and short (-1) signals based on MFI
    for i in range(1, len(signals)):
        if signals['MFI'][i] > 80:  # You can adjust this overbought threshold as needed
            signals.loc[i, 'mfi_signal'] = 'overbought'  # short signal (MFI is overbought)
        elif signals['MFI'][i] < 20:  # You can adjust this oversold threshold as needed
            signals.loc[i, 'mfi_signal'] = 'oversold' # long signal (MFI is oversold)

    signals.drop(['MFI'], axis=1, inplace=True)

    return signals

# %%
# EOM
def eom_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    signals['eom_signal'] = 0

    # Calculate Ease of Movement (EOM) Indicator
    eom = volume.EaseOfMovementIndicator(stock_df['High'], stock_df['Low'], stock_df['Volume'], window)

    signals['EOM'] = eom.ease_of_movement()

    # Generate bullish (1) and bearish (-1) signals based on EOM
    for i in range(1, len(signals)):
        if signals['EOM'][i] > 0:  # You can adjust this threshold as needed
            signals.loc[i, 'eom_signal'] = 'bullish'  # Bullish signal (EOM is positive)
        elif signals['EOM'][i] < 0:  # You can adjust this threshold as needed
            signals.loc[i, 'eom_signal'] = 'bearish'  # Bearish signal (EOM is negative)

    signals.drop(['EOM'], axis=1, inplace=True)

    return signals


#%%
#RSI Divergence strategy

# def rsi_signals_with_divergence(stock_data, window=14, long_threshold=30, short_threshold=70, width=10):
#     signals = pd.DataFrame(index=stock_data.index)
#     signals['RSI_signal'] = 0  # Initialize the signal column with zeros

#     # Calculate RSI
#     rsi = momentum.RSIIndicator(stock_data['Close'], window)

#     signals['RSI'] = rsi.rsi()

#%%
# ADX 

#NEEDS WORK

# def adx_strength_direction(stock_df, window=14):
#     # Create ADX indicator
#     adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

#     # Create a DataFrame to store the results
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['ADX'] = adx.adx()  # Call the adx method to get the ADX values
#     signals['adx_Trend_Strength'] = 0
#     signals['adx_Direction'] = 0

#     # Determine trend strength and direction based on ADX values
#     for i in range(window, len(signals)):
#         if signals['ADX'][i] > 25:
#             signals.loc[signals.index[i], 'adx_Trend_Strength'] = 'strong'
#         elif signals['ADX'][i] < 20:
#             signals.loc[signals.index[i], 'adx_Trend_Strength'] = 'weak'

#         if adx.adx_pos()[i] > adx.adx_neg()[i]:  # Call the adx_pos and adx_neg methods
#             signals.loc[signals.index[i], 'adx_Direction'] = 'bullish'
#         elif adx.adx_pos()[i] < adx.adx_neg()[i]:  # Call the adx_pos and adx_neg methods
#             signals.loc[signals.index[i], 'adx_Direction'] = 'bearish'

#     return signals

# # %%
# # Mass Index

# # NEEDS WORK

# def mass_index_signals(stock_data, fast_window=9, slow_window=25):
#     # Create a DataFrame to store the trading signals
#     signals = pd.DataFrame(index=stock_data.index)
#     signals['Signal'] = 0

#     # Calculate Mass Index
#     mass_index = trend.MassIndex(stock_data['High'], stock_data['Low'], fast_window, slow_window)
    
#     # Calculate Short and Long EMAs
#     short_ema = trend.EMAIndicator(stock_data['Close'], window=fast_window)
#     long_ema = trend.EMAIndicator(stock_data['Close'], window=slow_window)

#     # Calculate the Mass Index and its reversal thresholds
#     mass_index_values = mass_index.mass_index()
#     reversal_bulge_threshold = 27
#     reversal_bulge_exit_threshold = 26.50

#     # Generate trading signals
#     in_downtrend = short_ema.ema_indicator() > long_ema.ema_indicator()

#     for i in range(len(signals)):
#         if in_downtrend[i] is True and mass_index_values[i] > reversal_bulge_threshold and mass_index_values[i - 1] <= reversal_bulge_exit_threshold:
#             signals.loc[signals.index[i], 'Signal'] = 'long'
#         elif in_downtrend[i] is False and mass_index_values[i] > reversal_bulge_threshold and mass_index_values[i - 1] <= reversal_bulge_exit_threshold:
#             signals.loc[signals.index[i], 'Signal'] = 'short'

#     return signals

# # %%
# # PSAR 
 
# # NEEDS WORK

# def psar_signals(stock_df, step=0.02, max_step=0.2):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['psar_direction'] = ''
#     signals['psar_signal'] = ''

#     # Calculate Parabolic SAR (PSAR)
#     psar = trend.PSARIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], step, max_step)
#     psar = psar.psar_down_indicator

#     for i in range(1, len(signals)):
#         if stock_df['Close'][i] > psar[i] and stock_df['Close'][i - 1] <= psar[i - 1]:
#             signals.loc[signals.index[i], 'psar_signal'] = 'long'  # Bullish crossover (long signal)
#         elif stock_df['Close'][i] < psar[i] and stock_df['Close'][i - 1] >= psar[i - 1]:
#             signals.loc[signals.index[i], 'psar_signal'] = 'short'  # Bearish crossover (short signal)

#         if stock_df['Close'][i] > psar[i] :
#             signals.loc[signals.index[i], 'psar_direction'] = 'bullish'
#         elif stock_df['Close'][i] < psar[i] :
#             signals.loc[signals.index[i], 'psar_direction'] = 'bearish'

#     return signals

# csv_file = 'SPY.csv'
# stock_df = pd.read_csv(csv_file)
# ao_signals = pd.DataFrame(psar_signals(stock_df))
# print(ao_signals)


# # %%
# # STC

# #NEEDS WORK

# def stc_signals(stock_df, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['stc_signal'] = 0
#     signals['stc_direction'] = 0

#     # Calculate STC (Stochastic RSI)
#     stc = trend.STCIndicator(stock_df['Close'], window_slow, window_fast, cycle, smooth1, smooth2)

#     # Determine overbought/oversold conditions
#     overbought_condition = stc > 75
#     oversold_condition = stc < 25

#     # Determine bullish/bearish trend
#     bullish_condition = stc > 50
#     bearish_condition = stc < 50

#     # Generate signals and directions
#     signals.loc[overbought_condition, 'stc_signal'] = 'overbought'
#     signals.loc[oversold_condition, 'stc_signal'] = 'oversold'

#     signals.loc[bullish_condition, 'stc_direction'] = 'Bullish'
#     signals.loc[bearish_condition, 'stc_direction'] = 'Bearish'


#     return signals

# # %%
# # Vortex

# # NEEDS WORK

# def vortex_signals(stock_df, window=14, threshold=1.0):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['vortex_signal'] = 0
#     signals['vortex_direction_signal'] = 0

#     # Calculate the Vortex Indicator
#     vortex = trend.VortexIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

#     # Create signals based on the crossover of the positive and negative vortex indicators
#     signals['Positive'] = vortex.vortex_indicator_pos
#     signals['Negative'] = vortex.vortex_indicator_neg

#     for i in range(1, len(signals)):
#         if signals['Positive'][i] > signals['Negative'][i] and signals['Positive'][i - 1] <= signals['Negative'][i - 1]:
#             signals.loc[signals.index[i], 'vortex_signal'] = 'long'  # long signal
#         elif signals['Positive'][i] < signals['Negative'][i] and signals['Positive'][i - 1] >= signals['Negative'][i - 1]:
#             signals.loc[signals.index[i], 'vortex_signal'] = 'short'  # short signal

#         if signals['Positive'][i] > signals['Negative'][i] :
#             signals.loc[signals.index[i], 'vortex_direction_signal'] = 'bullish'  # long signal
#         elif signals['Positive'][i] < signals['Negative'][i]:
#             signals.loc[signals.index[i], 'vortex_direction_signal'] = 'bearish'

#     return signals

# # %%
# # Weighted Moving Average
# # Golden Cross WMA

# # NEEDS WORK

# def golden_wma_signals(stock_df, short_period=50, long_period=200):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['wma_direction'] = 0
#     signals['wma_signal'] = 0

#     # Calculate short and long SMAs
#     short_wma = trend.WMAIndicator(stock_df['Close'], short_period)
#     long_wma = trend.WMAIndicator(stock_df['Close'], long_period)

#     # Determine market direction (Bullish or Bearish)
#     signals.loc[short_wma.wma() > long_wma.wma, 'wma_direction'] = 'bullish'
#     signals.loc[short_wma.wma() <= long_wma.wma(), 'mwa_direction'] = 'bearish'

#     # Generate long and short signals
#     signals.loc[(short_wma.wma() > long_wma.wma()) &
#                 (short_wma.wma().shift(1) <= long_wma.wma().shift(1)), 'wma_signal'] = 'long'
#     signals.loc[(short_wma.wma() <= long_wma.wma()) &
#                 (short_wma.wma().shift(1) > long_wma.wma().shift(1)), 'wma_signal'] = 'short'

#     return signals

# #%%
# # 13-26 MA STrategy

# # NEEDS WORK

# def short_wma_signals(stock_df, short_period=13, long_period=26):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['wma_direction'] = 0
#     signals['wma_signal'] = 0

#     # Calculate short and long SMAs
#     short_wma = trend.WMAIndicator(stock_df['Close'], short_period)
#     long_wma = trend.WMAIndicator(stock_df['Close'], long_period)

#     # Determine market direction (Bullish or Bearish)
#     signals.loc[short_wma.wma() > long_wma.wma, 'wma_direction'] = 'bullish'
#     signals.loc[short_wma.wma() <= long_wma.wma(), 'wma_direction'] = 'bearish'

#     # Generate long and short signals
#     signals.loc[(short_wma.wma() > long_wma.wma()) &
#                 (short_wma.wma().shift(1) <= long_wma.wma().shift(1)), 'wma_signal'] = 'long'
#     signals.loc[(short_wma.wma() <= long_wma.wma()) &
#                 (short_wma.wma().shift(1) > long_wma.wma().shift(1)), 'wma_signal'] = 'short'

#     return signals

# # %%
# # Donchain Channel

# # NEEDS WORK

# def donchian_channel_strategy(stock_df, window=20):
#     signals = pd.DataFrame(index=stock_df.index)
#     signals['dc_signal'] = 0

#     # Calculate Donchian Channel
#     donchian = volatility.DonchianChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

#     # Determine long and short signals
#     for i in range(window, len(signals)):
#         upper_channel = donchian.donchian_channel_hband()[i]
#         lower_channel = donchian.donchian_channel_lband()[i]
#         current_close = stock_df['Close'][i]

#         if current_close > upper_channel:
#             signals.loc[signals.index[i], 'dc_signal'] = 'long'  # long signal (breakout above upper channel)
#         elif current_close < lower_channel:
#             signals.loc[signals.index[i], 'dc_signal'] = 'short'  # short signal (breakout below lower channel)

#     return signals