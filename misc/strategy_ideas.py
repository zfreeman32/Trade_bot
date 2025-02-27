    
#%%
import pandas as pd 
import numpy as np 
from ta import trend, volatility, volume, momentum, trend

#%%
# Functions from elegantoscillatorstrat.py
def rms(data):
    """Calculates the root mean square (RMS) for a given series."""
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise ValueError("data must be a Pandas Series, NumPy array, or list.")
    
    data = pd.to_numeric(data, errors='coerce').ffill()  # Keep it as a Pandas Series
    data = data.to_numpy()  # Convert only after ffill()

    if len(data) == 0 or np.all(np.isnan(data)):
        return np.nan
    
    return np.sqrt(np.mean(data**2))  # Ensure RMS returns a single float value

def supersmoother(data, length=10):
    """Applies the SuperSmoother filter to a Pandas Series."""
    if not isinstance(data, (pd.Series, np.ndarray, list)):
        raise ValueError("data must be a Pandas Series, NumPy array, or list.")
    
    data = pd.to_numeric(data, errors='coerce').ffill()  # Keep as Pandas Series
    
    if len(data) < length or np.all(np.isnan(data)):
        return pd.Series(np.full(len(data), np.nan), index=data.index)
    
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c2, c3 = -b1, a1 * a1
    c1 = 1 - c2 - c3
    
    ss = np.zeros(len(data))
    ss[:2] = data[:2]  # Initialize first two values

    for i in range(2, len(data)):
        ss[i] = c1 * (data.iloc[i] + data.iloc[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    
    return pd.Series(ss, index=data.index)

def elegant_oscillator(stock_df, rms_length=10, cutoff_length=10, threshold=0.5):
    """Computes the Elegant Oscillator and generates buy/sell signals."""
    if 'Close' not in stock_df:
        raise ValueError("stock_df must contain a 'Close' column.")
    
    close_price = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()
    
    if close_price.isna().all():
        raise ValueError("All values in 'Close' are NaN after conversion.")
    
    # Compute RMS on a rolling window correctly
    rms_values = close_price.rolling(rms_length).apply(lambda x: rms(x), raw=False)

    if rms_values.isna().all():
        ss_filter = pd.Series(np.nan, index=close_price.index)
    else:
        ss_filter = supersmoother(rms_values, cutoff_length)
    
    if ss_filter.isna().all():
        elegant_oscillator_values = pd.Series(np.nan, index=close_price.index)
    else:
        min_ss, max_ss = ss_filter.min(), ss_filter.max()
        if min_ss == max_ss:
            elegant_oscillator_values = pd.Series(0, index=close_price.index)
        else:
            x = (2 * (ss_filter - min_ss) / (max_ss - min_ss) - 1)
            elegant_oscillator_values = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    
    # Initialize buy/sell signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['elegant_oscillator_buy_signal'] = 0
    signals['elegant_oscillator_sell_signal'] = 0
    
    # Generate signals
    signals.loc[(elegant_oscillator_values > threshold) & 
                (elegant_oscillator_values.shift(1) <= threshold), 'elegant_oscillator_buy_signal'] = 1
    signals.loc[(elegant_oscillator_values < -threshold) & 
                (elegant_oscillator_values.shift(1) >= -threshold), 'elegant_oscillator_sell_signal'] = 1
    
    return signals

#%%
# Functions from ertrend.py
# Compute Efficiency Ratio
def compute_ER(data, window=14):
    """Compute Efficiency Ratio (ER) while ensuring numeric input."""
    data = pd.to_numeric(data['Close'], errors='coerce').ffill()
    change = data.diff()
    volatility = change.abs().rolling(window, min_periods=1).sum()  # Ensure small datasets work
    ER = change.rolling(window, min_periods=1).sum()
    
    # Avoid division by zero
    ER = np.where(ER == 0, np.nan, volatility / ER)

    return pd.Series(ER, index=data.index)

def ERTrend_signals(stock_df, ER_window=14, ER_avg_length=14, lag=7, avg_length=14, 
                    trend_level=0.5, max_level=1.0, crit_level=0.2, mult=1.5, average_type='simple'):
    """Computes buy/sell signals based on the Efficiency Ratio (ER) trend."""
    signals = pd.DataFrame(index=stock_df.index)

    # Ensure 'Close' is numeric
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()
    
    # Compute moving average
    if average_type == 'simple':
        MA = trend.SMAIndicator(stock_df['Close'], avg_length).sma_indicator()
    else:
        MA = stock_df['Close'].ewm(span=avg_length).mean()
    
    # Compute Efficiency Ratio (ER)
    ER = compute_ER(stock_df['Close'], ER_window)

    # Ensure all Series are aligned with stock_df index
    MA = MA.reindex(stock_df.index)
    ER = ER.reindex(stock_df.index)
    
    lowest_ER = ER.rolling(lag, min_periods=1).min().reindex(stock_df.index)
    highest_ER = ER.rolling(lag, min_periods=1).max().reindex(stock_df.index)
    
    # Generate buy signal conditions
    buy_flag = ((ER > crit_level) & (ER > lowest_ER * mult) & (stock_df['Close'] > MA)).fillna(False)
    strong_trend = ((ER > trend_level) & (ER < max_level)).fillna(False)

    # Initialize buy/sell signal columns
    signals['ERTrend_buy_signal'] = 0
    signals['ERTrend_sell_signal'] = 0
    
    # Set buy signal where conditions are met
    signals.loc[buy_flag & strong_trend, 'ERTrend_buy_signal'] = 1
    
    # Generate sell signal conditions
    sell_close_condition = (stock_df['Close'].shift(-1) < MA) & (stock_df['Close'].shift(-2) > MA)
    sell_close_condition = sell_close_condition.fillna(False)

    # Set sell signal where conditions are met
    signals.loc[sell_close_condition, 'ERTrend_sell_signal'] = 1
    
    return signals

# Functions from firsthourbreakout.py

import pytz

def FirstHourBreakout(data):
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize(pytz.utc)
    data.index = data.index.tz_convert(pytz.timezone('US/Eastern'))
    
    signals = pd.DataFrame(index=data.index)
    signals['FirstHourBreakout_buy_signal'] = 0
    signals['FirstHourBreakout_sell_signal'] = 0
    
    market_open = pd.Timestamp('09:30', tz='US/Eastern').time()
    first_hour_end = pd.Timestamp('10:30', tz='US/Eastern').time()
    market_close = pd.Timestamp('16:15', tz='US/Eastern').time()
    
    grouped = data.groupby(data.index.normalize())
    
    for day, day_data in grouped:
        if day_data.empty:
            continue
        
        first_hour_data = day_data.between_time(market_open, first_hour_end)
        if first_hour_data.empty:
            continue
        
        first_hour_high = first_hour_data['High'].max()
        first_hour_low = first_hour_data['Low'].min()
        
        signals.loc[day_data['High'] > first_hour_high, 'FirstHourBreakout_buy_signal'] = 1
        signals.loc[day_data['Low'] < first_hour_low, 'FirstHourBreakout_sell_signal'] = 1
        
    return signals
# Functions from meanreversionswingle.py


import talib 


def detect_uptrend(close_prices, min_length=20, min_range_for_uptrend=5.0):
    """Detects an uptrend by checking if the rolling high-low range is greater than a threshold."""
    
    if not isinstance(close_prices, pd.Series):
        raise ValueError("close_prices must be a Pandas Series.")

    close_prices = pd.to_numeric(close_prices, errors='coerce').ffill()

    rolling_min = close_prices.rolling(window=min_length, min_periods=1).min()
    rolling_max = close_prices.rolling(window=min_length, min_periods=1).max()

    return (rolling_max - rolling_min) > min_range_for_uptrend

def detect_pullback(close_prices, tolerance=1.0):
    """Detects a pullback by checking if the price is below its recent high by a given tolerance."""
    
    if not isinstance(close_prices, pd.Series):
        raise ValueError("close_prices must be a Pandas Series.")
    
    close_prices = pd.to_numeric(close_prices, errors='coerce').ffill()
    rolling_high = close_prices.rolling(window=10, min_periods=1).max()
    
    return (rolling_high - close_prices) > tolerance

def detect_up_move(close_prices, pullback_signal, min_up_move=0.5):
    """Detects an upward move after a pullback."""
    
    if not isinstance(close_prices, pd.Series):
        raise ValueError("close_prices must be a Pandas Series.")
    
    if not isinstance(pullback_signal, pd.Series):
        pullback_signal = pd.Series(pullback_signal, index=close_prices.index)

    close_prices = pd.to_numeric(close_prices, errors='coerce').ffill()
    price_diff = close_prices.diff()

    return (price_diff > min_up_move) & pullback_signal

def mean_reversion_swing_le(stock_df, min_length=20, max_length=400, min_range_for_uptrend=5.0,
                            min_up_move=0.5, tolerance=1.0):
    """Computes the Mean Reversion Swing strategy buy signals."""
    
    signals = pd.DataFrame(index=stock_df.index)

    # Ensure 'Close' column exists and is numeric
    if 'Close' not in stock_df or stock_df['Close'].isnull().all():
        raise ValueError("Stock data must contain a 'Close' column with at least some valid numeric values.")

    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    # Ensure that detect functions receive a Pandas Series
    close_prices = stock_df['Close'].copy()

    # Compute trend-based conditions safely
    signals['uptrend'] = detect_uptrend(close_prices, min_length, min_range_for_uptrend)
    
    # Detect pullbacks correctly
    signals['pullback'] = detect_pullback(close_prices, tolerance)

    # Ensure `signals['pullback']` exists before calling detect_up_move()
    signals['up_move'] = detect_up_move(close_prices, signals['pullback'], min_up_move)

    # Generate trade signals
    signals['mean_reversion_swing_buy_signal'] = np.where(signals['uptrend'] & signals['pullback'] & signals['up_move'], 1, 0)
    signals['mean_reversion_swing_sell_signal'] = 0  # No sell logic implemented

    # Ensure the pattern length does not exceed max_length
    signals = limit_pattern_length(signals, max_length)

    # Drop intermediate calculation columns
    signals = signals.drop(columns=['uptrend', 'pullback', 'up_move'])

    return signals

# Functions from middlehighlowmastrat.py



def calculate_mid_range(price_high, price_low):
    return (price_high + price_low) / 2

def calculate_moving_average(data, length, average_type='simple'):
    # Ensure data is a numeric 1D Series
    data = pd.to_numeric(data, errors='coerce').dropna()

    if average_type == 'simple':
        return data.rolling(window=length).mean()
    elif average_type == 'exponential':
        return data.ewm(span=length).mean()
    elif average_type == 'weighted':
        if len(data) < length:
            return pd.Series(np.nan, index=data.index)  # Prevents short-length errors
        weights = np.arange(1, length + 1)  # Fix array length
        return data.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    elif average_type == 'wilder':
        return data.ewm(alpha=1/length).mean()
    elif average_type == 'hull':
        sqrt_length = int(np.sqrt(length))  # Ensure valid window size
        if sqrt_length < 1:
            return pd.Series(np.nan, index=data.index)
        return np.sqrt(data.rolling(window=sqrt_length).mean()).ewm(span=sqrt_length).mean()
    else:
        raise ValueError("Invalid average type. Expected one of: 'simple', 'exponential', 'weighted', 'wilder', 'hull'")

def MHLMA_signals(stock_df, length=50, high_low_length=15, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Ensure numeric conversion before calculations
    stock_df[['High', 'Low', 'Close']] = stock_df[['High', 'Low', 'Close']].apply(pd.to_numeric, errors='coerce')

    # Compute MidRange with correct input arguments
    rolling_high = stock_df['High'].rolling(window=high_low_length).max()
    rolling_low = stock_df['Low'].rolling(window=high_low_length).min()
    
    stock_df['MidRange'] = calculate_mid_range(rolling_high, rolling_low)

    # Ensure MidRange is numeric
    stock_df['MidRange'] = pd.to_numeric(stock_df['MidRange'], errors='coerce')

    # Compute moving averages
    signals['MA'] = calculate_moving_average(stock_df['Close'], length, average_type)
    signals['MHL_MA'] = calculate_moving_average(stock_df['MidRange'], length, average_type)
    
    # Generate signals
    signals['MHLMA_buy_signal'] = 0
    signals['MHLMA_sell_signal'] = 0
    signals.loc[signals['MA'] > signals['MHL_MA'], 'MHLMA_buy_signal'] = 1
    signals.loc[signals['MA'] < signals['MHL_MA'], 'MHLMA_sell_signal'] = 1

    # Drop intermediate columns
    signals = signals.drop(['MA'], axis=1)

    return signals

# Functions from r2trend.py


from scipy.stats import linregress
import statsmodels.api as sm

def calculate_RSquared(series, length=18):
    """Calculate R-Squared of a rolling linear regression."""
    series = pd.Series(series).dropna()  # Convert to Series and drop NaNs

    if len(series) < length:
        return np.nan  # Prevent errors when there isn't enough data

    X = np.arange(length).reshape(-1, 1)  # Ensure X is 1D
    Y = series.values[-length:].astype(float)  # Ensure Y is numeric

    if len(X) != len(Y):  # Ensure dimensions match
        return np.nan

    model = sm.OLS(Y, sm.add_constant(X))  # Fit model
    results = model.fit()
    return results.rsquared


def calculate_slope(series, length=18):
    """Calculate the slope of a rolling linear regression."""
    series = pd.Series(series).dropna()  # Convert to Series and drop NaNs

    if len(series) < length:
        return np.nan  # Prevent errors when there isn't enough data

    X = np.arange(length).reshape(-1, 1)  # Ensure X is a 1D array
    Y = series.values[-length:].astype(float)  # Ensure Y is 1D and numeric

    if len(X) != len(Y):  # Ensure dimensions match
        return np.nan

    return linregress(X.flatten(), Y).slope  # Ensure 1D input


def calculate_moving_average(df, length=18, average_type='simple'):
    df = pd.to_numeric(df, errors='coerce')  # Ensure numeric data
    if len(df) < length:
        return np.nan  # Prevent errors when there isn't enough data

    if average_type == 'simple':
        return df.rolling(length).mean().iloc[-1]  # Use iloc[-1] safely
    elif average_type == 'exponential':
        return df.ewm(span=length).mean().iloc[-1]  # Use iloc[-1] safely
    elif average_type == 'weighted':
        return df.rolling(length).apply(lambda x: np.average(x, weights=np.arange(1, len(x) + 1))).iloc[-1]
    else:
        return np.nan  # If average type is unknown

def r2trend_signals(df, length=18, lag=10, average_length=50, max_level=0.85, lr_crit_level=10, average_type='SIMPLE'):
    signals = pd.DataFrame(index=df.index)

    # Ensure Close is numeric before processing
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce').ffill()

    signals['RSquared'] = df['Close'].rolling(length).apply(
        lambda x: calculate_RSquared(pd.Series(x), length) if len(x) == length else np.nan, raw=False
    )
    
    signals['slope'] = df['Close'].rolling(length).apply(
        lambda x: calculate_slope(pd.Series(x), length) if len(x) == length else np.nan, raw=False
    )

    # Apply moving average calculation safely
    signals['ma'] = df['Close'].rolling(average_length).apply(
        lambda x: calculate_moving_average(pd.Series(x), average_length, average_type) if len(x) == average_length else np.nan,
        raw=False
    )

    # Fill missing values in ma to prevent NaN comparison errors
    signals['ma'].fillna(method='ffill', inplace=True)

    signals['RSquared_lag'] = signals['RSquared'].shift(lag)
    signals['r2trend_buy_signal'] = 0
    signals['r2trend_sell_signal'] = 0

    # Strong Uptrend Condition
    cond1 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] > lr_crit_level) & \
            (df['Close'] > signals['ma'])

    signals.loc[cond1, 'r2trend_buy_signal'] = 1

    # Strong Downtrend Condition
    cond2 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] < -lr_crit_level) & \
            (df['Close'] < signals['ma'])

    signals.loc[cond2, 'r2trend_sell_signal'] = 1
    signals.drop(['slope', 'ma', 'RSquared_lag'], axis=1, inplace=True)

    return signals

# Functions from reverseemastrat.py



def reverse_ema(price_series, period):
    """Computes a weighted rolling percentage change (Z-transform)."""
    price_series = pd.to_numeric(price_series, errors='coerce')  # Ensure numeric
    z_price = price_series.pct_change()  # Compute percentage change
    weights = np.arange(1, period + 1)  # Generate weights from 1 to period
    return z_price.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def reverse_ema_strat(df, trend_length=39, cycle_length=6):
    signals = pd.DataFrame(index=df.index)

    # Ensure Close column is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Compute trend and cycle EMAs with correct period passing
    trend_ema = reverse_ema(df['Close'], trend_length)
    cycle_ema = reverse_ema(df['Close'], cycle_length)

    buy_signal_col = 'reverse_ema_strat_buy_signal'
    sell_signal_col = 'reverse_ema_strat_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    # Generate signals
    signals.loc[(cycle_ema > 0) & (trend_ema > 0), buy_signal_col] = 1
    signals.loc[(cycle_ema < 0) & (trend_ema < 0), sell_signal_col] = 1

    return signals

# Functions from rsitrend.py



from scipy.signal import argrelextrema

# Custom ZigZag Function
def calculate_zigzag(price_series, percentage_reversal=5):
    """
    Calculates the ZigZag indicator based on price changes.
    :param price_series: The price series (Close prices).
    :param percentage_reversal: Percentage move required to define a ZigZag.
    :return: ZigZag trend values (+1 for peaks, -1 for troughs, 0 otherwise)
    """
    price_series = pd.to_numeric(price_series, errors='coerce').ffill()  # Ensure numeric and fill NaN
    zigzag = np.zeros(len(price_series))

    if len(price_series) < 5:  # Ensure enough data
        return zigzag  # Return an array of zeros if not enough data

    # Calculate percentage price move
    price_change = price_series.pct_change() * 100

    # Set a dynamic `order` value (default 2, but lower for small datasets)
    order = min(2, len(price_series) // 3)

    # Find local maxima (peaks) with correct comparator
    peaks = argrelextrema(price_series.values, comparator=np.greater, order=order)[0]
    for peak in peaks:
        if peak < len(price_change) and price_change.iloc[peak] >= percentage_reversal:
            zigzag[peak] = 1  # Mark as peak

    # Find local minima (troughs) with correct comparator
    troughs = argrelextrema(price_series.values, comparator=np.less, order=order)[0]
    for trough in troughs:
        if trough < len(price_change) and abs(price_change.iloc[trough]) >= percentage_reversal:
            zigzag[trough] = -1  # Mark as trough

    return zigzag

# RSI Trend Strategy with ZigZag
def rsi_trend_signals(stock_df, length=14, over_bought=70, over_sold=30, percentage_reversal=5, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    price = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()  # Ensure numeric data

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
    signals['rsi_trend_buy_signal'] = 0
    signals['rsi_trend_sell_signal'] = 0
    signals.loc[conditions_buy, 'rsi_trend_buy_signal'] = 1
    signals.loc[conditions_sell, 'rsi_trend_sell_signal'] = 1

    return signals

# Functions from simplemeanreversion.py



def calculate_z_score(series):
    """Calculate the Z-score of a series, ensuring numerical consistency."""
    series = pd.to_numeric(series, errors='coerce').dropna()  # Convert to numeric and drop NaNs
    mean = series.mean()
    std_dev = np.nanstd(series)  # Use nanstd to avoid division errors

    if std_dev == 0:  # Prevent divide-by-zero errors
        return pd.Series(np.nan, index=series.index)

    return (series - mean) / std_dev


def simple_moving_average(series, window):
    """Compute a simple moving average with error handling."""
    series = pd.to_numeric(series, errors='coerce').dropna()  # Ensure numeric
    return series.rolling(window).mean()


def simple_mean_reversion_signals(stock_df, length=30, fast_length_factor=1, slow_length_factor=10,
                                  z_score_entry_level=1.0, z_score_exit_level=0.5):

    # Ensure Close column is numeric before processing
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    close_prices = stock_df['Close']
    signals = pd.DataFrame(index=stock_df.index)

    # Compute SMA and Z-Score
    fast_window = max(1, length * fast_length_factor)  # Ensure valid window size
    slow_window = max(1, length * slow_length_factor)  # Ensure valid window size

    signals['FasterSMA'] = simple_moving_average(close_prices, fast_window)
    signals['SlowerSMA'] = simple_moving_average(close_prices, slow_window)
    signals['zScore'] = calculate_z_score(close_prices)  # Now safe to calculate

    # Initialize signal column
    signals['simple_mean_reversion_buy_signal'] = 0
    signals['simple_mean_reversion_sell_signal'] = 0

    # Buy Signal: When Z-Score < -Entry Level and Faster SMA > Slower SMA
    signals.loc[(signals['zScore'] < -z_score_entry_level) & (signals['FasterSMA'] > signals['SlowerSMA']), 'simple_mean_reversion_buy_signal'] = 1

    # Sell Signal: When Z-Score > Entry Level and Faster SMA < Slower SMA
    signals.loc[(signals['zScore'] > z_score_entry_level) & (signals['FasterSMA'] < signals['SlowerSMA']), 'simple_mean_reversion_sell_signal'] = 1

    # Drop unnecessary columns
    signals = signals[['simple_mean_reversion_buy_signal', 'simple_mean_reversion_sell_signal']]

    return signals





def calculate_moving_avg(close, length=10, average_type='simple'):
    """Calculates different types of moving averages."""
    
    if not isinstance(close, (pd.Series, np.ndarray, list)):
        raise ValueError("Input 'close' must be a Pandas Series, NumPy array, or list.")
    
    close = pd.to_numeric(close, errors='coerce').ffill()  # Ensure numeric data
    
    if len(close) < length:
        return pd.Series(np.nan, index=close.index)  # Return NaNs if insufficient data
    
    if average_type == 'simple':
        return close.rolling(window=length).mean()
    elif average_type == 'exponential':
        return close.ewm(span=length, adjust=False).mean()
    elif average_type == 'wilder':
        return trend.EMAIndicator(close=close, window=length, fillna=False).ema_indicator()
    elif average_type == 'hull':
        return trend.WMAIndicator(close=close, window=length, fillna=False).wma_indicator()
    else:
        raise ValueError("Invalid 'average_type'. Choose from 'simple', 'exponential', 'wilder', or 'hull'.")

def trend_following_signals(stock_df, length=10, average_type='simple', entry_percent=3.0, exit_percent=4.0):
    """Generates buy/sell signals using a trend-following strategy."""
    
    if 'Close' not in stock_df:
        raise ValueError("stock_df must contain a 'Close' column.")
    
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    if stock_df['Close'].isna().all():  # Ensure there's valid data
        raise ValueError("All values in 'Close' are NaN after conversion.")

    close = stock_df['Close'].copy()

    # Compute moving average
    moving_avg = calculate_moving_avg(close, length, average_type)
    
    if moving_avg.isna().all():
        raise ValueError(f"Moving average calculation failed for type '{average_type}'.")

    # Define conditions
    long_entry = close > moving_avg * (1 + entry_percent / 100)
    long_exit = close < moving_avg * (1 - exit_percent / 100)
    short_entry = close < moving_avg * (1 - entry_percent / 100)
    short_exit = close > moving_avg * (1 + exit_percent / 100)

    # Initialize buy/sell signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['trend_following_buy_signal'] = 0
    signals['trend_following_sell_signal'] = 0

    # Assign 'buy' signals
    signals.loc[long_entry | short_exit, 'trend_following_buy_signal'] = 1

    # Assign 'sell' signals
    signals.loc[long_exit | short_entry, 'trend_following_sell_signal'] = 1

    return signals

# Functions from universaloscillatorstrat.py


def ehlers_universal_oscillator(close, cutoff_length=20):
    alpha = (np.cos(np.sqrt(2) * np.pi / cutoff_length))**2
    beta = 1 - alpha
    a = (1 - beta / 2) * beta
    b = (1 - alpha) * (1 - beta / 2)
    filt = np.zeros_like(close)
    for i in range(2, close.shape[0]):
        filt[i] = a * (close[i] - 2*close[i-1] + close[i-2]) + 2*alpha*filt[i-1] - (alpha**2)*filt[i-2]
    osc = (filt - np.mean(filt)) / np.std(filt)
    return osc

def universal_oscillator_strategy(data, cutoff_length=20, mode='reversal'):
    signals = pd.DataFrame(index=data.index)
    close = np.array(data['Close'])
    osc = ehlers_universal_oscillator(close, cutoff_length)
    signals['osc'] = osc
    signals['universal_oscillator_strategy_buy_signal'] = 0
    signals['universal_oscillator_strategy_sell_signal'] = 0

    if mode == 'trend_following':
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_buy_signal'] = 1
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_sell_signal'] = 1
    elif mode == 'reversal':
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_buy_signal'] = 1
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_sell_signal'] = 1
    
    signals.drop(['osc'], axis=1, inplace=True)
    return signals

def frama_signals(stock_df, base_length=20, upper_limit=8, lower_limit=40, atr_length=14, atr_multiplier=1.9):
    """
    Implements the G-FRAMA | QuantEdgeB strategy.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close'.
    - base_length: Base length for FRAMA calculation.
    - upper_limit: Minimum smoothing length.
    - lower_limit: Maximum smoothing length.
    - atr_length: ATR period.
    - atr_multiplier: Multiplier for ATR-based filtering.

    Returns:
    - A DataFrame containing 'frama_signal' column with 'buy' or 'sell' signals.
    """

    signals = pd.DataFrame(index=stock_df.index)

    # === Compute FRAMA ===
    high_max = stock_df['High'].rolling(base_length).max()
    low_min = stock_df['Low'].rolling(base_length).min()
    hl_range = (high_max - low_min) / base_length

    half_length = base_length // 2
    high_max1 = stock_df['High'].rolling(half_length).max()
    low_min1 = stock_df['Low'].rolling(half_length).min()
    high_max2 = stock_df['High'].shift(half_length).rolling(half_length).max()
    low_min2 = stock_df['Low'].shift(half_length).rolling(half_length).min()

    hl1 = (high_max1 - low_min1) / half_length
    hl2 = (high_max2 - low_min2) / half_length

    D = np.log(hl1 + hl2) - np.log(hl_range)
    D /= np.log(2)
    
    # Fix: Proper Series assignment
    D.loc[hl1 <= 0] = D.shift(1).fillna(D)
    D.loc[hl2 <= 0] = D.shift(1).fillna(D)

    w = np.log(2 / (lower_limit + 1))
    alpha = np.exp(w * (D - 1))
    alpha = np.clip(alpha, 0.01, 1)

    old_N = (2 - alpha) / alpha
    new_N = (lower_limit - upper_limit) * (old_N - 1) / (lower_limit - 1) + upper_limit
    new_alpha = 2 / (new_N + 1)
    new_alpha = np.clip(new_alpha, 2 / (lower_limit + 1), 1)

    stock_df['FRAMA'] = stock_df['Close'].ewm(alpha=new_alpha, adjust=False).mean()

    # === Compute ATR Filter ===
    atr = volatility.AverageTrueRange(stock_df['High'], stock_df['Low'], stock_df['Close'], window=atr_length)
    stock_df['ATR'] = atr.average_true_range() * atr_multiplier  

    stock_df['LongV'] = stock_df['FRAMA'] + stock_df['ATR']
    stock_df['ShortV'] = stock_df['FRAMA'] - stock_df['ATR']

    # === Generate 'buy' and 'sell' Signals ===
    stock_df['frama_signal'] = 'neutral'
    stock_df.loc[stock_df['Close'] > stock_df['LongV'], 'frama_signal'] = 'buy'
    stock_df.loc[stock_df['Close'] < stock_df['ShortV'], 'frama_signal'] = 'sell'

    # Store only the 'buy'/'sell' signal column
    signals['frama_signal'] = stock_df['frama_signal']

    return signals

def fractal_ema_signals(stock_df, ema_short=10, ema_medium=20, ema_long=100, n=2):
    """
    Implements the Forex Fractal EMA Scalper strategy.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close'.
    - ema_short: Period for the short EMA.
    - ema_medium: Period for the medium EMA.
    - ema_long: Period for the long EMA.
    - n: Number of periods for fractal detection.

    Returns:
    - A DataFrame containing 'fractal_ema_signal' column with 'buy' or 'sell' signals.
    """

    signals = pd.DataFrame(index=stock_df.index)

    # === Compute EMAs ===
    stock_df['EMA_10'] = trend.EMAIndicator(stock_df['Close'], ema_short).ema_indicator()
    stock_df['EMA_20'] = trend.EMAIndicator(stock_df['Close'], ema_medium).ema_indicator()
    stock_df['EMA_100'] = trend.EMAIndicator(stock_df['Close'], ema_long).ema_indicator()

    # === Compute Up Fractal ===
    up_fractal = np.full(len(stock_df), False)
    for i in range(n, len(stock_df) - n):
        up_frontier = all(stock_df['High'].iloc[i - j] < stock_df['High'].iloc[i] for j in range(1, n+1))
        up_backfrontier = any(stock_df['High'].iloc[i + j] < stock_df['High'].iloc[i] for j in range(1, n+1))
        up_fractal[i] = up_frontier and up_backfrontier

    stock_df['Up_Fractal'] = up_fractal

    # === Compute Down Fractal ===
    down_fractal = np.full(len(stock_df), False)
    for i in range(n, len(stock_df) - n):
        down_frontier = all(stock_df['Low'].iloc[i - j] > stock_df['Low'].iloc[i] for j in range(1, n+1))
        down_backfrontier = any(stock_df['Low'].iloc[i + j] > stock_df['Low'].iloc[i] for j in range(1, n+1))
        down_fractal[i] = down_frontier and down_backfrontier

    stock_df['Down_Fractal'] = down_fractal

    # === Generate 'buy' and 'sell' Signals ===
    stock_df['fractal_ema_sell_signal'] = 0
    stock_df['fractal_ema_buy_signal'] = 0
    stock_df.loc[(stock_df['Up_Fractal']) & (stock_df['EMA_10'] > stock_df['EMA_20']) & (stock_df['EMA_20'] > stock_df['EMA_100']), 'fractal_ema_buy_signal'] = 1
    stock_df.loc[(stock_df['Down_Fractal']) & (stock_df['EMA_10'] < stock_df['EMA_20']) & (stock_df['EMA_20'] < stock_df['EMA_100']), 'fractal_ema_sell_signal'] = 1

    # Store only the 'buy'/'sell' signal column
    signals = stock_df['fractal_ema_sell_signal', 'fractal_ema_buy_signal']

    return signals


# Functions from onsettrend.py


def onset_trend_detector(stock_df, k1=0.8, k2=0.4):
    # Placeholder implementations
    def super_smoother_filter(data):
        pass  # Replace with actual implementation
    def roofing_filter(data):
        pass  # Replace with actual implementation
    def quotient_transform(data, k):
        pass  # Replace with actual implementation

    signals = pd.DataFrame(index=stock_df.index)
    
    # Apply Super Smoother Filter and Roofing Filter
    filtered_price = roofing_filter(super_smoother_filter(stock_df['Close']))
    
    # Compute oscillators using quotient transform
    oscillator1 = quotient_transform(filtered_price, k1)
    oscillator2 = quotient_transform(filtered_price, k2)
    
    # Generate signals
    signals['oscillator1'] = oscillator1
    signals['oscillator2'] = oscillator2
    signals['onset_trend_detector_buy_signal'] = 0
    signals['onset_trend_detector_sell_signal'] = 0
    signals.loc[(signals['oscillator1'] > 0) & (signals['oscillator1'].shift(1) <= 0), 'onset_trend_detector_buy_signal'] = 1
    signals.loc[(signals['oscillator2'] < 0) & (signals['oscillator2'].shift(1) >= 0), 'onset_trend_detector_sell_signal'] = 1
    
    signals.drop(['oscillator1', 'oscillator2'], axis=1, inplace=True)
    
    return signals

