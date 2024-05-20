#%%
from typing import Self
import pandas as pd
from ta import volatility
from ta import trend
import matplotlib.pyplot as plt

#%%
def atr_signals(stock_df, atr_window=14, ema_window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['trend_strength'] = ''

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

csv_file = 'SPY.csv'
stock_df = pd.read_csv(csv_file)
ao_signals = pd.DataFrame(atr_signals(stock_df))
print(ao_signals)


# %%
# Donchain Channel
def donchian_channel_strategy(stock_df, window=20):
    signals = pd.DataFrame(index=stock_df.index)
    signals['dc_signal'] = 0


    # Calculate Donchian Channel
    donchian = volatility.DonchianChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window)

    # Determine buy and sell signals
    for i in range(window, len(signals)):
        upper_channel = donchian.donchian_channel_hband()[i]
        lower_channel = donchian.donchian_channel_lband()[i]
        current_close = stock_df['Close'][i]

        if current_close > upper_channel:
            signals.loc[signals.index[i], 'dc_signal'] = 'buy'  # Buy signal (breakout above upper channel)
        elif current_close < lower_channel:
            signals.loc[signals.index[i], 'dc_signal'] = 'sell'  # Sell signal (breakout below lower channel)

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

# %%
