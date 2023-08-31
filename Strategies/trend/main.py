#%%
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

csv_file = r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot\SPY.csv'
stock_df = pd.read_csv(csv_file)
ao_signals = pd.DataFrame(dpo_signals(stock_df))
print(ao_signals)
# %%
# EMA
def ema_signals(stock_df, short_window=12, long_window=26):
    signals = pd.DataFrame(index=stock_df.index)
    signals['EMA_Signal'] = 0  # Initialize the signal column with zeros

    # Calculate short-term EMA
    ema_short = stock_df['Close'].ewm(span=short_window, adjust=False).mean()

    # Calculate long-term EMA
    ema_long = stock_df['Close'].ewm(span=long_window, adjust=False).mean()

    # Generate EMA signals
    for i in range(1, len(stock_df)):
        if ema_short[i] > ema_long[i] and ema_short[i - 1] <= ema_long[i - 1]:
            signals.loc[stock_df.index[i], 'EMA_Signal'] = 1  # Bullish (Buy) Signal
        elif ema_short[i] < ema_long[i] and ema_short[i - 1] >= ema_long[i - 1]:
            signals.loc[stock_df.index[i], 'EMA_Signal'] = -1  # Bearish (Sell) Signal

    return signals