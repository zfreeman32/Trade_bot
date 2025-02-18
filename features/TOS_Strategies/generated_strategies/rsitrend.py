import pandas as pd
import numpy as np
from ta import momentum
from scipy.signal import argrelextrema

# Custom ZigZag Function
def calculate_zigzag(price_series, percentage_reversal=5):
    """
    Calculates the ZigZag indicator based on price changes.
    :param price_series: The price series (Close prices).
    :param percentage_reversal: Percentage move required to define a ZigZag.
    :return: ZigZag trend values (+1 for peaks, -1 for troughs, 0 otherwise)
    """
    zigzag = np.zeros(len(price_series))

    # Calculate percentage price move
    price_change = price_series.pct_change() * 100

    # Find local maxima (peaks)
    peaks = argrelextrema(price_series.values, np.greater, order=2)[0]
    for peak in peaks:
        if price_change.iloc[peak] >= percentage_reversal:
            zigzag[peak] = 1  # Mark as peak

    # Find local minima (troughs)
    troughs = argrelextrema(price_series.values, np.less, order=2)[0]
    for trough in troughs:
        if abs(price_change.iloc[trough]) >= percentage_reversal:
            zigzag[trough] = -1  # Mark as trough

    return zigzag

# RSI Trend Strategy with ZigZag
def rsi_trend_signals(stock_df, length=14, over_bought=70, over_sold=30, percentage_reversal=5, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    price = stock_df['Close']

    # Compute RSI
    if average_type == 'simple':
        rsi = momentum.RSIIndicator(price, length)
    elif average_type == 'exponential':
        rsi = momentum.RSIIndicator(price.ewm(span=length).mean(), length)
    elif average_type == 'weighted':
        weights = np.arange(1, length + 1)
        rsi = momentum.RSIIndicator((price * weights).sum() / weights.sum(), length)

    # Compute ZigZag Trend
    zigzag_values = calculate_zigzag(price, percentage_reversal)

    # Define Buy and Sell Conditions
    conditions_buy = (rsi.rsi() > over_sold) & (zigzag_values > 0)
    conditions_sell = (rsi.rsi() < over_bought) & (zigzag_values < 0)

    # Assign signals
    signals['rsi_trend_signal'] = 'neutral'
    signals.loc[conditions_buy, 'rsi_trend_signal'] = 'buy'
    signals.loc[conditions_sell, 'rsi_trend_signal'] = 'sell'

    return signals
