import numpy as np
import pandas as pd


def ATR(data, period=14):
    true_ranges = []
    atr = 0

    for i in range(1, len(data)):
        high = data[i]['high']
        low = data[i]['low']
        prev_close = data[i - 1]['close']
        true_range = max(high - low, abs(high - prev_close),
                            abs(low - prev_close))
        true_ranges.append(true_range)

        if len(true_ranges) > period:
            true_ranges = true_ranges[1:]

        atr = sum(true_ranges) / len(true_ranges)
    return pd.DataFrame(atr)

def macd(data, fast_period=12, slow_period=26, signal_period=9):
    exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram})

def RSI(data, period=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return pd.DataFrame(rsi)

def ADX(data, period=14):
    dm_pos = data['High'].diff().where(lambda x: x > 0, 0)
    dm_neg = -data['Low'].diff().where(lambda x: x < 0, 0)
    atr = pd.concat([(dm_pos + dm_neg).ewm(span=period, adjust=False).mean(),
                    abs(data['High'] - data['Close'].shift()
                        ).ewm(span=period, adjust=False).mean(),
                    abs(data['Low'] - data['Close'].shift()).ewm(span=period, adjust=False).mean()], axis=1).max(axis=1)
    di_pos = 100 * dm_pos.ewm(span=period, adjust=False).mean() / atr
    di_neg = 100 * dm_neg.ewm(span=period, adjust=False).mean() / atr
    adx = 100 * abs(di_pos - di_neg) / (di_pos + di_neg)
    return pd.DataFrame(adx)

def TrailingStop(prices, percentage=.05):
    stop_values = []
    high_price = prices[0]

    for price in prices:
        if price > high_price:
            high_price = price
        stop_value = high_price * (1 - percentage)
        stop_values.append(stop_value)

    return pd.DataFrame(stop_values)

def EMA(data):
    ema = data['Close'].ewm(span=20, adjust=False).mean()
    return ema

def pivot_points(data):
    pivot = (data['High'] + data['Low'] + data['Close']) / 3
    s1 = (2 * pivot) - data['High']
    s2 = pivot - (data['High'] - data['Low'])
    r1 = (2 * pivot) - data['Low']
    r2 = pivot + (data['High'] - data['Low'])
    return pd.DataFrame({'pivot': pivot, 'support_1': s1, 'support_2': s2, 'resistance_1': r1, 'resistance_2': r2})

def stochastic_oscillator(data, n=14):
    high = data['High'].rolling(n).max()
    low = data['Low'].rolling(n).min()
    k = 100 * ((data['Close'] - low) / (high - low))
    d = k.rolling(3).mean()
    return pd.DataFrame({'%K': k, '%D': d})

def vwap(data):
    v = data['Volume']
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (v * tp).cumsum() / v.cumsum()
    return pd.DataFrame(vwap)

def money_flow_index(data, n=14):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    mf_raw = tp * data['Volume']
    mf_positive = mf_raw.where(tp.diff() > 0, 0)
    mf_negative = mf_raw.where(tp.diff() < 0, 0)
    mf_ratio = mf_positive.rolling(n).sum() / mf_negative.rolling(n).sum()
    mfi = 100 - (100 / (1 + mf_ratio))
    return pd.DataFrame(mfi)
