#%%
from ta import add_all_ta_features
from ta import momentum
import numpy as np
import pandas as pd
# from scipy.signal import argrelextrema
from collections import deque

#%%
# Higher Highs
def getHigherHighs(data: np.array, order=5, K=2):
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i-1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

# Lower Lows
def getLowerLows(data: np.array, order=5, K=2):
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i-1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

# Higher Lows
def getHigherLows(data: np.array, order=5, K=2):
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if lows[i] > lows[i-1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema

# Lower Highs
def getLowerHighs(data: np.array, order=5, K=2):
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    extrema = []
    ex_deque = deque(maxlen=K)
    
    for i, idx in enumerate(high_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
        if highs[i] < highs[i-1]:
            ex_deque.clear()
        ex_deque.append(idx)
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())

    return extrema
#%%
# PPO
def ppo_signals(stock_data, fast_window=12, slow_window=26, signal_window=9):
    signals = pd.DataFrame(index=stock_data.index)
    signals['PPO_signal'] = 0

    ppo = momentum.PercentagePriceOscillator(stock_data['Close'], fast_window, slow_window, signal_window)

    ppo_values = ppo.ppo()
    ppo_signal = ppo.ppo_signal()

    # Generate buy (1) and sell (-1) signals based on PPO and its signal line
    for i in range(1, len(stock_data)):
        if ppo_values[i] > ppo_signal[i] and ppo_values[i - 1] <= ppo_signal[i - 1]:
            signals.loc[stock_data.index[i], 'PPO_signal'] = ['buy']  # Buy signal (PPO crosses above signal line)
        elif ppo_values[i] < ppo_signal[i] and ppo_values[i - 1] >= ppo_signal[i - 1]:
            signals.loc[stock_data.index[i], 'PPO_signal'] = ['sell']  # Sell signal (PPO crosses below signal line)

    return signals

#%%
# Awesome Oscilator 0 cross
def Awesome_Oscillator_signals(stock_df):
    # Define buy and sell signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Ao_signal'] = 0  # Initialize all signals to 0

    for i in range(1, len(stock_df)):
        if (
            stock_df['momentum_ao'].iloc[i-1] < 0 and
            stock_df['momentum_ao'].iloc[i] >= 0
        ):
            signals['Ao_signal'].iloc[i] = ['buy']  # Buy signal
        elif (
            stock_df['momentum_ao'].iloc[i-1] >= 0 and
            stock_df['momentum_ao'].iloc[i] < 0
        ):
            signals['Ao_signal'].iloc[i] = ['sell']  # Sell signal
    return signals

#%%
# KAMA Cross
def kama_cross_signals(stock_df, fast_period=10, slow_period=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['kama_signal'] = 0

    # Calculate Fast KAMA
    fast_kama = momentum.kama(stock_df['Close'], window=fast_period)

    # Calculate Slow KAMA
    slow_kama = momentum.kama(stock_df['Close'], window=slow_period)
    for i in range(1, len(stock_df)):
        if fast_kama[i] > slow_kama[i] and fast_kama[i - 1] <= slow_kama[i - 1] and stock_df['Close'][i] > fast_kama[i]:
            signals['kama_signal'].iloc[i] = ['buy']  # Buy signal (fast KAMA above slow KAMA and price above fast KAMA)
        elif fast_kama[i] < slow_kama[i] and fast_kama[i - 1] >= slow_kama[i - 1] and stock_df['Close'][i] < fast_kama[i]:
            signals['kama_signal'].iloc[i] = ['sell']  # Sell signal (fast KAMA below slow KAMA and price below fast KAMA)
        elif stock_df['Close'][i] > fast_kama[i] and stock_df['Close'][i - 1] <= fast_kama[i - 1]:
            stock_df.loc[i, 'Signal'] = ['buy']  # Buy signal (price crosses above KAMA)
        elif stock_df['Close'][i] < fast_kama[i] and stock_df['Close'][i - 1] >= fast_kama[i - 1]:
            stock_df.loc[i, 'Signal'] = ['sell']

    return signals

#%%
#RSI Divergence strategy

# def rsi_signals_with_divergence(stock_data, window=14, buy_threshold=30, sell_threshold=70, width=10):
#     signals = pd.DataFrame(index=stock_data.index)
#     signals['RSI_signal'] = 0  # Initialize the signal column with zeros

#     # Calculate RSI
#     rsi = momentum.RSIIndicator(stock_data['Close'], window)

#     signals['RSI'] = rsi.rsi()

#%%

def stoch_signals(stock_df, fast_period=10, slow_period=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stoch_signal'] = 0

    # Calculate Stochastic Oscillator
    stoch = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=14, smooth_window=3)

    signals['%K'] = stoch.stoch()
    signals['%D'] = stoch.stoch_signal()

    # Generate buy (1) and sell (-1) signals based on Stochastic Oscillator
    for i in range(1, len(signals)):
        if signals['%K'][i] > signals['%D'][i] and signals['%K'][i - 1] <= signals['%D'][i - 1]:
            signals.loc[i, 'stoch_signal'] = ['buy']  # Buy signal (%K crosses above %D)
        elif signals['%K'][i] < signals['%D'][i] and signals['%K'][i - 1] >= signals['%D'][i - 1]:
            signals.loc[i, 'stoch_signal'] = ['sell']  # Sell signal (%K crosses below %D)

    return signals

#%%
# TSI 
def tsi_signals(stock_df, window_slow=25, window_fast=13):
    signals = pd.DataFrame(index=stock_df.index)
    signals['tsi_signal'] = 0

    # Calculate True Strength Index (TSI)
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)

    signals['TSI'] = tsi.tsi()

    # Generate buy (1) and sell (-1) signals based on TSI
    for i in range(1, len(signals)):
        if signals['TSI'][i] > 0 and signals['TSI'][i - 1] <= 0:
            signals.loc[i, 'tsi_signal'] = ['buy']  # Buy signal (TSI crosses above 0)
        elif signals['TSI'][i] < 0 and signals['TSI'][i - 1] >= 0:
            signals.loc[i, 'tsi_signal'] = ['sell']  # Sell signal (TSI crosses below 0)

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
            signals.loc[i, 'williams_signal'] = ['overbought']  # Overbought signal (Williams %R crosses below or equal to -80)
        elif signals['WilliamsR'][i] >= -20:
            signals.loc[i, 'williams_signal'] = ['oversold']  # Oversold signal (Williams %R crosses above or equal to -20)

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
            signals.loc[i, 'roc_signal'] = ['overbought']  # Overbought signal (ROC crosses above or equal to 10)
        elif signals['ROC'][i] <= -10:
            signals.loc[i, 'roc_signal'] = ['oversold']  # Oversold signal (ROC crosses below or equal to -10)

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
            signals.loc[i, 'rsi_signal'] = ['overbought']  # Overbought signal (RSI crosses above or equal to 70)
        elif signals['RSI'][i] <= 30:
            signals.loc[i, 'rsi_signal'] = ['oversold']  # Oversold signal (RSI crosses below or equal to 30)

    return signals
# %%
def stochrsi_signals(stock_df, window=14, smooth1=3, smooth2=3):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stochrsi_signal'] = 0

    # Calculate StochRSI
    stoch_rsi = momentum.StochRSIIndicator(stock_df['Close'], window, smooth1, smooth2)

    signals['StochRSI'] = stoch_rsi.stochrsi()

    # Generate overbought (1) and oversold (-1) signals based on StochRSI
    for i in range(1, len(signals)):
        if signals['StochRSI'][i] >= 0.8:
            signals.loc[i, 'stochrsi_signal'] = ['overbought']  # Overbought signal (StochRSI crosses above or equal to 0.8)
        elif signals['StochRSI'][i] <= 0.2:
            signals.loc[i, 'stochrsi_signal'] = ['oversold']  # Oversold signal (StochRSI crosses below or equal to 0.2)

    return signals

#%%
