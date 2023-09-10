
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf

# Define the stock symbol and the timeframe
symbol = 'SPY'
interval = '30m'

# Retrieve the stock price data
data = yf.download(tickers = "SPY",  # list of tickers
            period = "60d",         # time period
            interval = "30m",       # trading interval
            prepost = False,       # download pre/post market hours data?
            repair = True)    
# Read data from CSV file
# data = pd.read_csv(r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot\SPY.csv')


# In[2]

# Value > Avg and
# ADX[1] < ADX and
# RSI[1] < RSI and
# price[1] < trail[1] and
# price > trail


def calculate_trailing_stop(prices, percentage):
    stop_values = []
    high_price = prices[0]

    for price in prices:
        if price > high_price:
            high_price = price
        stop_value = high_price * (1 - percentage)
        stop_values.append(stop_value)

    return stop_values


def calculate_adx(high, low, close, n=14):
    true_range = pd.DataFrame(
        {'A': high - low, 'B': abs(high - close.shift()), 'C': abs(low - close.shift())})
    true_range = true_range.max(axis=1)
    true_range_avg = true_range.rolling(n).mean()

    directional_movement = pd.DataFrame(
        {'Up': high.diff(), 'Down': -low.diff()})
    positive_directional_movement = (
        directional_movement['Up'] > directional_movement['Down']) & (directional_movement['Up'] > 0)
    negative_directional_movement = (directional_movement['Down'] > directional_movement['Up']) & (
        directional_movement['Down'] > 0)
    positive_directional_index = (
        positive_directional_movement * directional_movement['Up']).rolling(n).sum()
    negative_directional_index = (
        negative_directional_movement * directional_movement['Down']).rolling(n).sum()

    true_range_avg[true_range_avg == 0] = 0.0001
    positive_directional_index_avg = positive_directional_index.rolling(
        n).mean() / true_range_avg
    negative_directional_index_avg = negative_directional_index.rolling(
        n).mean() / true_range_avg

    adx = (abs(positive_directional_index_avg - negative_directional_index_avg) /
           (positive_directional_index_avg + negative_directional_index_avg)) * 100
    return adx


def calculate_rsi(prices, n=14):
    deltas = prices.diff()
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum() / n
    down = -seed[seed < 0].sum() / n
    rs = up / down
    rsi = pd.Series(index=prices.index)
    rsi[n] = 100 - (100 / (1 + rs))

    for i in range(n + 1, len(prices)):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0
        else:
            upval = 0
            downval = -delta

        up = (up * (n - 1) + upval) / n
        down = (down * (n - 1) + downval) / n
        rs = up / down
        rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(prices, span):
    ema_values = []
    alpha = 2 / (span + 1)
    ema_prev = prices[0]

    ema_values.append(ema_prev)

    for price in prices[1:]:
        ema = (price - ema_prev) * alpha + ema_prev
        ema_values.append(ema)
        ema_prev = ema

    return ema_values



def calculate_deviation_percentile(data):
    data['EMA'] = data['Close'].ewm(span=100).mean()  # Calculate 100-EMA
    data['Deviation'] = data['Close'] - data['EMA']  # Calculate deviation

    percentiles = []
    for i in range(len(data)):
        if i >= 500:
            # Calculate the 95th percentile of the past 500 points
            deviation_tail = data['Deviation'].iloc[i-500:i]
            percentile_95 = np.nanpercentile(deviation_tail, 95)
        else:
            # Not enough data points, set the percentile to NaN
            percentile_95 = np.nan

        percentiles.append(percentile_95)

    # Create a dataframe with the percentile values
    percentile_df = pd.DataFrame({'95th Percentile': percentiles})

    return percentile_df


def calculate_atr_trailing_stop(data, period=14, multiplier=3):
    # Calculate True Range
    data['High-Low'] = data['High'] - data['Low']
    data['High-Close'] = abs(data['High'] - data['Close'].shift())
    data['Low-Close'] = abs(data['Low'] - data['Close'].shift())
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)
    
    # Calculate Average True Range (ATR)
    data['ATR'] = data['True Range'].rolling(period).mean()
    
    # Calculate ATR trailing stop
    data['ATR Trailing Stop'] = data['Close'] - (multiplier * data['ATR'])
    
    # Remove intermediate columns
    data.drop(['High-Low', 'High-Close', 'Low-Close', 'True Range'], axis=1, inplace=True)
    
    return data['ATR Trailing Stop']

#In[4]
def calculate_result(data):
    ema_span = 10
    trailing_percentage = 0.05

    data['EMA'] = calculate_ema(data['Close'], ema_span)
    data['ADX'] = calculate_adx(data['High'], data['Low'], data['Close'])
    data['RSI'] = calculate_rsi(data['Close'])
    data['Trailing_Stop'] = calculate_trailing_stop(data['Close'], trailing_percentage)
    data['Deviation'] = data['Close'] - data['EMA']
    data['Percentile'] = calculate_deviation_percentile(data)
    data['ATR'] = calculate_atr_trailing_stop(data)
    return data

data = calculate_result(data)

def remove_first_500_rows(df):
    df = df.iloc[500:]  # Remove the first 500 rows
    df.reset_index(drop=True, inplace=True)  # Reset the index
    return df

# Example usage
data = remove_first_500_rows(data)

#In[2]
# Value > EMA and ADX[n-1] < ADX[n] and RSI[n-1] < RSI[n] and Close[n-1] < Trail[n-1] and Close[n] < Trail[n]
def define_buy_signal(data):
    buy_signal = (data['Close'] > data['EMA']) & (data['ADX'].shift() < data['ADX']) & \
                 (data['RSI'].shift() < data['RSI']) & (data['Close'].shift() < data['Trailing_Stop'].shift()) & \
                 (data['Close'] < data['Trailing_Stop']) & (data['Deviation'] < data['Percentile'])
    return buy_signal


def backtest_strategy(data): 
    buy_signal = define_buy_signal(data)
    positions = pd.Series(index=data.index, dtype=np.float64)
    positions[buy_signal] = 1.0  # Buy signal
    positions.ffill(inplace=True)  # Forward fill positions

    # Initialize lists to store trade indices and prices
    trade_indices = []
    trade_prices = []

    # Backtest the strategy
    position = 0
    pl = []
    buy_price = 0
    trailing_stop = 0
    for i in range(len(data)):
        if buy_signal[i] and position == 0:
            # Buy signal
            position = 1
            buy_price = data['Close'][i]
            trailing_stop = buy_price * 0.95  # Set initial trailing stop
            pl.append(0)  # No P/L at entry
            trade_indices.append(i)
            trade_prices.append(data['Close'][i])
        elif not buy_signal[i] and position == 1:
            # Sell signal based on trailing stop
            if data['Close'][i] < trailing_stop:
                position = 0
                pl.append(data['Close'][i-1] - buy_price)  # Capture P/L at exit
                trade_indices.append(i)
                trade_prices.append(data['Close'][i])
            else:
                # Update trailing stop
                trailing_stop = max(trailing_stop, data['Close'][i] * 0.95)
                pl.append(0)  # No P/L
        else:
            # No trade or holding a position
            pl.append(0)  # No P/L

    # Calculate cumulative P/L
    cumulative_pl = pd.Series(pl).cumsum()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label='Close')
    plt.scatter(data.index[trade_indices], data['Close'].iloc[trade_indices], marker='^', color='g', label='Trade')
    plt.plot(data.index, cumulative_pl, label='Cumulative P/L')
    plt.xlabel('Date')
    plt.ylabel('Price / P/L')
    plt.title('EUREKA Strategy Backtest')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
backtest_strategy(data)



# %%
