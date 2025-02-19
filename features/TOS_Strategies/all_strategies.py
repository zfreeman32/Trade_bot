#%% Combined functions from multiple files

# Functions from accumulationdistributionstrat.py
#%%
import pandas as pd
import numpy as np
from ta import volatility

def acc_dist_strat(df, length = 4, factor = 0.75, vol_ratio= 1 , vol_avg_length = 4, mode='high-low'):
    
    signals = pd.DataFrame(index=df.index)
    
    if mode == 'high-low':
        df['range'] = df['High'] - df['Low']
    else:
        average_true_range = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=length)
        df['range'] = average_true_range.average_true_range()
        
    df['last_range'] = df['range'].shift(length)
    df['range_ratio'] = df['range'] / df['last_range']
    
    df['vol_av'] = df['Volume'].rolling(window=vol_avg_length).mean()
    df['vol_last_av'] = df['vol_av'].shift(vol_avg_length)
    df['vol_ratio_av'] = df['vol_av'] / df['vol_last_av']
    
    df['higher_low'] = df['Low'] > df['Low'].shift().rolling(window=12).min()
    df['break_high'] = df['Close'] > df['High'].shift().rolling(window=12).max()
    df['fall_below'] = df['Close'] < df['Low'].shift().rolling(window=12).min()
    
    # Buy signals
    signals['acc_dist_strat_buy'] = np.where(
        (df['range_ratio'] < factor) &
        df['higher_low'] &
        df['break_high'] &
        (df['vol_ratio_av'] > vol_ratio),
        1, 0)
    
    # Sell signals
    signals['acc_dist_strat_sell'] = np.where(
        df['fall_below'],
        1, 0)
    return signals

# Functions from adxbreakoutsle.py
import pandas as pd
from ta import trend

def adx_breakouts_signals(stock_df, highest_length=15, adx_length=14, adx_level=40, offset=0.5):
    signals = pd.DataFrame(index=stock_df.index)  # Ensure index matches stock_df
    
    highest = stock_df['High'].rolling(window=highest_length).max()
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_length).adx()

    signals['adx'] = adx
    signals['highest'] = highest
    signals['adx_breakout_buy_signal'] = 0

    breakout_condition = (signals['adx'] > adx_level) & (stock_df['Close'] > (signals['highest'] + offset))
    
    # Ensure index consistency
    breakout_condition = breakout_condition.reindex(stock_df.index, fill_value=False)

    signals.loc[breakout_condition, 'adx_breakout_buy_signal'] = 1
    signals.drop(['adx', 'highest'], axis=1, inplace=True)

    return signals

# Functions from adxtrend.py
import pandas as pd
import numpy as np
from ta import trend

def ADXTrend_signals(stock_df, length=14, lag=14, average_length=14, trend_level=25, max_level=50, crit_level=20, mult=2, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    close = stock_df['Close']
    
    # Calculate ADX
    adx_indicator = trend.ADXIndicator(high=stock_df['High'], low=stock_df['Low'], close=close, window=length)
    signals['ADX'] = adx_indicator.adx()

    # Calculate minimum ADX value over 'lag' period
    signals['min_ADX'] = signals['ADX'].rolling(window=lag, min_periods=1).min()

    # Calculate moving average of 'close' prices
    if average_type == 'simple':
        signals['avg_price'] = close.rolling(window=average_length).mean()
    elif average_type == 'exponential':
        signals['avg_price'] = close.ewm(span=average_length, adjust=False).mean()
    elif average_type == 'weighted':
        weights = np.arange(1, average_length + 1)
        signals['avg_price'] = close.rolling(window=average_length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    # Generate signals according to strategy rules
    signals['adx_trend_buy_signal'] = np.where(
        (
            ((signals['ADX'] > signals['min_ADX'] * mult) & (signals['ADX'] > crit_level)) |
            ((signals['ADX'] > trend_level) & (signals['ADX'] < max_level))
        ) & (close > signals['avg_price']), 1, 0)
    signals['adx_trend_sell_signal'] = np.where(
            (
                ((signals['ADX'] > signals['min_ADX'] * mult) & (signals['ADX'] > crit_level)) |
                ((signals['ADX'] > trend_level) & (signals['ADX'] < max_level))
            ) & (close < signals['avg_price']), 1, 0)

    signals.drop(['ADX', 'min_ADX', 'avg_price'], axis=1, inplace=True)

    return signals

# Functions from atrhighsmabreakoutsle.py
import pandas as pd
from ta import trend, volatility

def atr_high_sma_breakouts_le(df, atr_period=14, sma_period=100, offset=.5, wide_range_candle=True, volume_increase=True):
    """
    ATRHighSMABreakoutsLE strategy by Ken Calhoun.

    df: pandas.DataFrame: OHLCV data.
    atr_period: int: Period for ATR calculation.
    sma_period: int: Period for SMA calculation.
    offset: float: 'Recorded high' + 'offset' is entry trigger.
    wide_range_candle: bool: True to only trigger if candle height is at least 1.5x the average.
    volume_increase: bool: True to only trigger if the volume has increased since the last candle.
    """

    # Calculate ATR and SMA
    atr = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
    sma = trend.SMAIndicator(df['Close'], window=sma_period).sma_indicator()

    # Create dataframe to store signals
    signals = pd.DataFrame(index=df.index)
    signals['recorded_high'] = df['High'].shift()*(df['High'].shift()==df['High'].rolling(window=atr_period).max())
    signals['current_candle_height'] = df['High'] - df['Low']
    signals['average_candle_height'] = signals['current_candle_height'].rolling(window=atr_period).mean()

    # Conditions
    condition1 = (atr == atr.rolling(window=atr_period).max()) & (df['Close'] > sma)
    condition2 = (df['High'] > (signals['recorded_high'] + offset))
    condition3 = (signals['current_candle_height'] > 1.5 * signals['average_candle_height']) if wide_range_candle else True
    condition4 = (df['Volume'] > df['Volume'].shift()) if volume_increase else True

    # Create signals
    signals['atr_high_sma_breakouts_le_buy_signal'] = 0
    signals.loc[condition1 & condition2 & condition3 & condition4, 'atr_high_sma_breakouts_le_buy_signal'] = 1
    signals.drop(['recorded_high'], axis=1, inplace=True)
    return signals


# Functions from atrtrailingstople.py
import pandas as pd
from ta import volatility,trend

def atr_trailing_stop_le_signals(stock_df, atr_period=14, atr_factor=3, average_type='simple'):
    close = stock_df['Close']
    high = stock_df['High']
    low = stock_df['Low']
    
    atr = volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
    atr_trailing_stop = close - atr_factor * atr

    if average_type == 'exponential':
        atr_trailing_stop = trend.EMAIndicator(atr_trailing_stop, atr_period).ema_indicator()
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['ATR Trailing Stop'] = atr_trailing_stop
    signals['atr_trailing_stop_le_buy_signal'] = np.where(
        (signals['close'].shift(1) <= signals['atr_trailing_stop'].shift(1)) &  # Previous close was below ATR
        (signals['close'] > signals['atr_trailing_stop']),  # Current close crosses above ATR
        1,  # Signal triggered
        0   # No signal
    )

    signals.drop(['ATR Trailing Stop'], axis=1, inplace=True)

    return signals

# Functions from atrtrailingstopse.py
import pandas as pd
import numpy as np

def atr_trailing_stop_se_signals(stock_df, atr_period=14, atr_factor=3.0, trail_type="unmodified"):

    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']

    # Depending on the trail type, use either unmodified ATR or a modified ATR calculation
    if trail_type == "unmodified":
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
    elif trail_type == "modified":
        atr = atr * atr_factor
    else:
        raise ValueError(f"Unsupported trail type '{trail_type}'")

    # Calculate the ATR Trailing Stop value
    atr_trailing_stop = close.shift() - atr

    # Create a DataFrame for the signals
    signals = pd.DataFrame({'atr_trailing_stop': atr_trailing_stop}, index=stock_df.index)
    signals['atr_se_sell_signal'] = np.where(
        (signals['close'].shift(1) >= signals['atr_trailing_stop']) &  # Previous close was above ATR
        (signals['close'] < signals['atr_trailing_stop']),  # Current close crosses below ATR
        1,  # Signal triggered
        0   # No signal
    )

    # Drop the internal 'atr_trailing_stop' column, as it is not part of the output
    signals.drop(columns=['atr_trailing_stop'], inplace=True)

    return signals


# Functions from bbdivergencestrat.py
import pandas as pd
import talib

def BB_Divergence_Strat(dataframe, secondary_data=None):
    signal = []
    high = dataframe['High']
    low = dataframe['Low']
    close = dataframe['Close']

    # Calculate all the necessary indicators
    dataframe['MADivergence3d'] = talib.MAX(dataframe['MADivergence'], timeperiod=3) if 'MADivergence' in dataframe else 0
    dataframe['MIDivergence3d'] = talib.MIN(dataframe['MIDivergence'], timeperiod=3) if 'MIDivergence' in dataframe else 0
    dataframe['ROC'] = talib.ROC(close, timeperiod=3)
    
    # Calculate MACD
    dataframe['MACD'], dataframe['MACDsignal'], dataframe['MACDhist'] = talib.MACD(close)
    
    # Calculate Stochastic
    dataframe['SOK'], dataframe['SOD'] = talib.STOCH(high, low, close)

    # Handle secondary data (if available)
    if secondary_data is not None:
        if 'MADivergence' in secondary_data and 'MIDivergence' in secondary_data:
            secondary_data['MADivergence3d'] = talib.MAX(secondary_data['MADivergence'], timeperiod=3)
            secondary_data['MIDivergence3d'] = talib.MIN(secondary_data['MIDivergence'], timeperiod=3)

    # Define Buy and Sell Conditions
    conditions_buy = (
        (dataframe['MADivergence3d'] > 20) &
        (dataframe['MACD'] > dataframe['MACDsignal']) &
        (dataframe['ROC'] > 0) &
        (dataframe['SOK'] < 85) 
    )
    
    conditions_sell = (
        (dataframe['MIDivergence3d'] < -20) &
        (dataframe['MACD'] < dataframe['MACDsignal']) &
        (dataframe['ROC'] < 0) &
        (dataframe['SOK'] > 85)
    )

    # Assign Buy and Sell signals
    dataframe['BB_Divergence_Strat_buy_signal'] = 0
    dataframe['BB_Divergence_Strat_sell_signal'] = 0
    dataframe.loc[conditions_buy, 'BB_Divergence_Strat_buy_signal'] = 1
    dataframe.loc[conditions_sell, 'BB_Divergence_Strat_sell_signal'] = 1
    
    return dataframe[['BB_Divergence_Strat_buy_signal', 'BB_Divergence_Strat_sell_signal']]

# Functions from bollingerbandsle.py
import pandas as pd
from ta import volatility

def bollinger_bands_le_signals(data, length=20, num_devs_dn=2.0):
    # Instantiate Bollinger Bands indicator
    indicator_bb = volatility.BollingerBands(close=data['Close'], window=length, window_dev=num_devs_dn)
    
    # Create DataFrame for signals
    signals = pd.DataFrame(index=data.index)
    signals['lower'] = indicator_bb.bollinger_lband()  # Calculate lower band
    signals['close'] = data['Close']  # Track closing prices
    signals['bollinger_bands_le_buy_signal'] = 0 # Default all signals to 0.0

    # Generate 'Long Entry' signal where price crosses above lower band
    signals['bollinger_bands_le_buy_signal'][signals['close'] > signals['lower'].shift(1)] = 1
    signals.drop(columns=['close', 'lower'], inplace=True)
    # Return only the signal column
    return signals

# Functions from bollingerbandsse.py
import pandas as pd
import numpy as np
from ta import volatility

def bollinger_bands_short_entry(df, length=20, num_devs_up=2):
    # Calculate Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=df["Close"], window=length, window_dev=num_devs_up)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    # Generate signals
    df['short_entry'] = np.where(df['Close'] < df['bb_bbh'], 1, 0)
    signals = pd.DataFrame(df['short_entry'])
    signals.columns = ['bb_short_entry_signal']
    return signals


# Functions from camarillapointsstrat.py
import pandas as pd

def camarilla_pivot_points(stock_df):
    high = stock_df['High']
    low = stock_df['Low']
    close = stock_df['Close']

    # Calculate the pivot point.
    pivot_point = (high + low + close) / 3

    # Calculate the support and resistance levels based on the pivot point.
    ranges = high - low
    r1 = close + ranges * 1.1 / 12
    s1 = close - ranges * 1.1 / 12
    r2 = close + ranges * 1.1 / 6
    s2 = close - ranges * 1.1 / 6
    r3 = close + ranges * 1.1 / 4
    s3 = close - ranges * 1.1 / 4
    r4 = close + ranges * 1.1 / 2
    s4 = close - ranges * 1.1 / 2

    stock_df['PP'] = pivot_point
    stock_df['R1'] = r1
    stock_df['R2'] = r2
    stock_df['R3'] = r3
    stock_df['R4'] = r4
    stock_df['S1'] = s1
    stock_df['S2'] = s2
    stock_df['S3'] = s3
    stock_df['S4'] = s4

    return stock_df

def camarilla_strategy(stock_df):
    stock_df = camarilla_pivot_points(stock_df)
    signals = pd.DataFrame(index=stock_df.index)
    signals['camarilla_buy_signal'] = 0
    signals['camarilla_sell_signal'] = 0
  
    # Long entry
    signals.loc[((stock_df.Open < stock_df.S3) & (stock_df.Close > stock_df.S3)), 'camarilla_buy_signal'] = 1
    # Short entry
    signals.loc[((stock_df.Open > stock_df.R3) & (stock_df.Close < stock_df.R3)), 'camarilla_sell_signal'] = 2

    return signals

# Functions from consbarsdownse.py
import numpy as np
import pandas as pd

def ConsBarsDownSE(data, consecutive_bars_down=4, price="Close"):
    """Generates a signal when a certain number of consecutive down bars are found.

    Args:
    data (pd.DataFrame): DataFrame containing OHLCV data.
    consecutive_bars_down (int, optional): Number of consecutive down bars to trigger the signal. Defaults to 4.
    price (str, optional): The price column to use in the analysis. Defaults to "Close".

    Returns:
    pd.DataFrame: DataFrame with signals.
    """
    signals = pd.DataFrame(index=data.index)

    # Create a boolean series of whether each bar is down
    bars_down = data[price] < data[price].shift(1)
    
    # Create a rolling sum of the down bars
    consecutive_sum = bars_down.rolling(window=consecutive_bars_down).sum()
    
    # Create a signal where the sum equals the specified number of consecutive bars
    signals["ConsBarsDownSE_sell_signal"] = np.where(consecutive_sum >= consecutive_bars_down, 1, 0)
    
    return signals

# Functions from consbarsuple.py
import pandas as pd

def cons_bars_up_le_signals(stock_df, consec_bars_up=4, price='Close'):
    # Initialize the signal DataFrame with the same index as stock_df
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0

    # Create a boolean mask indicating where price bars are consecutively increasing
    mask = stock_df[price].diff() > 0
    
    # Calculate the rolling sum of increasing bars
    rolling_sum = mask.rolling(window=consec_bars_up).sum()
    
    # Finally, as in template, long signals where the count of increasing bars exceeds consec_bars_up
    signals['Signal'][rolling_sum >= consec_bars_up] = 1

    # Take difference of signals to identify specific 'long' points
    signals['Signal'] = signals['Signal'].diff()

    # Replace any NaNs with 0
    signals['Signal'].fillna(0, inplace=True)

    # Generate trading orders
    signals['cons_bars_up_le_signal'] = 0
    signals.loc[signals['Signal'] > 0, 'cons_bars_up_le_buy_signal'] = 1
    signals.drop(columns=['Signal'], inplace=True)
    return signals

import pandas as pd
import numpy as np
from ta import volatility

def donchian_signals(df, entry_length=40, exit_length=15, atr_length=20, atr_factor=2, atr_stop_factor=2, atr_average_type='simple'):
    signals = pd.DataFrame(index=df.index)

    # Ensure numeric conversion
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Compute Donchian Channel Stops
    signals['BuyStop'] = df['High'].rolling(window=entry_length, min_periods=1).max().reindex(df.index)
    signals['ShortStop'] = df['Low'].rolling(window=entry_length, min_periods=1).min().reindex(df.index)
    signals['CoverStop'] = df['High'].rolling(window=exit_length, min_periods=1).max().reindex(df.index)
    signals['SellStop'] = df['Low'].rolling(window=exit_length, min_periods=1).min().reindex(df.index)

    # Compute ATR and reindex to match df
    atr = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_length).average_true_range().reindex(df.index)

    # Apply chosen smoothing type
    if atr_average_type == "simple":
        atr_smoothed = atr.rolling(window=atr_length, min_periods=1).mean().reindex(df.index)
    elif atr_average_type == "exponential":
        atr_smoothed = atr.ewm(span=atr_length, adjust=False).mean().reindex(df.index)
    elif atr_average_type == "wilder":
        atr_smoothed = atr.ewm(alpha=1/atr_length, adjust=False).mean().reindex(df.index)
    else:
        raise ValueError(f"Unsupported ATR smoothing type '{atr_average_type}'")

    # Preinitialize signal column
    signals['donchian_buy_signals'] = 0  # Default to 0 (neutral)
    signals['donchian_sell_signals'] = 0  # Default to 0 (neutral)

    # Apply ATR filter
    if atr_factor > 0:
        volatility_filter = atr_smoothed * atr_factor

        # Ensure proper alignment of boolean masks
        buy_condition = (df['High'] > signals['BuyStop']) & ((df['High'].shift(1) - volatility_filter) > 0)
        sell_condition = (df['Low'] < signals['ShortStop']) & ((df['Low'].shift(1) - volatility_filter) < 0)

        signals.loc[buy_condition, 'donchian_buy_signals'] = 1
        signals.loc[sell_condition, 'donchian_sell_signals'] = 1
    else:
        signals.loc[df['High'] > signals['BuyStop'], 'donchian_buy_signals'] = 1
        signals.loc[df['Low'] < signals['ShortStop'], 'donchian_sell_signals'] = 1

    # Drop unnecessary columns
    signals.drop(columns=['CoverStop', 'SellStop', 'BuyStop', 'ShortStop'], inplace=True)

    return signals.fillna(0)

# Functions from ehlersstoch.py
#%%
import pandas as pd
import numpy as np

# Ehlers Decycler Oscillator Signal Strategy
def ehlers_stoch_signals(stock_df, length1=10, length2=20):
    signals = pd.DataFrame(index=stock_df.index)
    price = stock_df['Close']
    
    # High-pass filter function
    def hp_filter(src, length):
        pi = 2 * np.arcsin(1)
        twoPiPrd = 0.707 * 2 * pi / length
        a = (np.cos(twoPiPrd) + np.sin(twoPiPrd) - 1) / np.cos(twoPiPrd)
        hp = np.zeros_like(src)
        for i in range(2, len(src)):
            hp[i] = ((1 - a / 2) ** 2) * (src[i] - 2 * src[i-1] + src[i-2]) + 2 * (1 - a) * hp[i-1] - ((1 - a) ** 2) * hp[i-2]
        return hp
    
    # Compute the Ehlers Decycler Oscillator
    hp1 = hp_filter(price, length1)
    hp2 = hp_filter(price, length2)
    dec = hp2 - hp1
    
    # Compute signal strength
    slo = dec - np.roll(dec, 1)
    sig = np.where(slo > 0, np.where(slo > np.roll(slo, 1), 2, 1),
                   np.where(slo < 0, np.where(slo < np.roll(slo, 1), -2, -1), 0))
    
    # Generate buy and sell signals with strong signals
    signals['ehlers_stoch_signals'] = 0
    signals.loc[sig == 1, 'ehlers_stoch_signals'] = 1
    signals.loc[sig == 2, 'ehlers_stoch_signals'] = 2
    signals.loc[sig == -1, 'ehlers_stoch_signals'] = 3
    signals.loc[sig == -2, 'ehlers_stoch_signals'] = 4
    
    return signals

# Functions from eightmonthavg.py
def eight_month_avg_signals(stock_df, length=8):
    signals = pd.DataFrame(index=stock_df.index)  # Ensure index matches stock_df
    signals['sma'] = stock_df['Close'].rolling(window=length).mean()
    signals['eight_month_avg_signals'] = 'neutral'

    buy_condition = (stock_df['Close'] > signals['sma']) & (stock_df['Close'].shift(1) <= signals['sma'].shift(1))
    sell_condition = (stock_df['Close'] < signals['sma']) & (stock_df['Close'].shift(1) >= signals['sma'].shift(1))

    # Ensure the conditions' index matches stock_df.index before applying
    buy_condition = buy_condition.reindex(stock_df.index, fill_value=False)
    sell_condition = sell_condition.reindex(stock_df.index, fill_value=False)

    signals.loc[buy_condition, 'eight_month_avg_signals'] = 1
    signals.loc[sell_condition, 'eight_month_avg_signals'] = 2

    signals.drop(['sma'], axis=1, inplace=True)
    return signals

# Functions from elegantoscillatorstrat.py
import pandas as pd
import numpy as np

def rms(data):
    """Calculates the root mean square (RMS)."""
    data = pd.to_numeric(data, errors='coerce').ffill()  # Ensure numeric and fill NaN
    return np.sqrt(np.mean(data**2, axis=0))

def supersmoother(data, length=10):
    """Applies the SuperSmoother filter to the input data."""
    data = pd.to_numeric(data, errors='coerce').ffill()  # Ensure numeric data

    if len(data) < length:  # Prevents issues if not enough data
        return np.full(len(data), np.nan)

    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(1.414 * np.pi / length)
    c2, c3 = -b1, a1 * a1
    c1 = 1 - c2 - c3

    ss = np.zeros_like(data)
    for i in range(2, len(data)):
        ss[i] = c1 * (data[i] + data[i-1]) / 2 + c2 * ss[i-1] + c3 * ss[i-2]
    
    return ss

def elegant_oscillator(stock_df, rms_length=10, cutoff_length=10, threshold=0.5):
    """Computes the Elegant Oscillator and generates buy/sell signals."""
    signals = pd.DataFrame(index=stock_df.index)

    # Ensure Close column is numeric
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    close_price = stock_df['Close']
    
    # Calculate Root Mean Square (RMS)
    stock_df['rms'] = rms(close_price)

    # Apply SuperSmoother filter
    stock_df['ss_filter'] = supersmoother(stock_df['rms'], cutoff_length)

    # Normalize and apply Inverse Fisher Transform
    min_ss, max_ss = np.nanmin(stock_df['ss_filter']), np.nanmax(stock_df['ss_filter'])

    # Prevent division by zero
    if min_ss == max_ss:
        stock_df['elegant_oscillator'] = 0
    else:
        x = (2 * (stock_df['ss_filter'] - min_ss) / (max_ss - min_ss) - 1)
        stock_df['elegant_oscillator'] = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    # Generate signals
    signals['elegant_oscillator_signal'] = 0
    signals.loc[(stock_df['elegant_oscillator'] > threshold) & 
                (stock_df['elegant_oscillator'].shift(1) <= threshold), 'elegant_oscillator_signal'] = 2
    signals.loc[(stock_df['elegant_oscillator'] < -threshold) & 
                (stock_df['elegant_oscillator'].shift(1) >= -threshold), 'elegant_oscillator_signal'] = 1

    return signals

# Functions from ertrend.py
import pandas as pd
import numpy as np
from ta import trend

# Compute Efficiency Ratio
def compute_ER(data, window=14):
    """Compute Efficiency Ratio (ER) while ensuring numeric input."""
    data = pd.to_numeric(data, errors='coerce').ffill()  # Ensure numeric values
    change = data.diff()  # Compute price change
    volatility = change.abs().rolling(window).sum()  # Compute volatility
    ER = change.rolling(window).sum()  # Compute ER as sum of absolute changes

    # Avoid division by zero errors
    ER = np.where(ER == 0, np.nan, volatility / ER)  

    return pd.Series(ER, index=data.index)  # Return Series with correct index

# ER Trend Signal Strategy
def ERTrend_signals(stock_df, ER_window=14, ER_avg_length=14, lag=7, avg_length=14, 
                    trend_level=0.5, max_level=1.0, crit_level=0.2, mult=1.5, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)

    # Ensure Close column is numeric
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    # Use selected moving average type
    if average_type == 'simple':
        MA = trend.SMAIndicator(stock_df['Close'], avg_length).sma_indicator()
    else:
        MA = stock_df['Close'].ewm(span=avg_length).mean()

    # Ensure MA has the same index as stock_df
    MA = MA.reindex(stock_df.index)

    # Compute ER and ensure it has the correct index
    ER = compute_ER(stock_df['Close'], ER_window)
    lowest_ER = ER.rolling(lag).min().reindex(stock_df.index)
    highest_ER = ER.rolling(lag).max().reindex(stock_df.index)

    # Ensure all conditions have the same index and valid boolean type
    buy_flag = ((ER > crit_level) & (ER > lowest_ER * mult) & (stock_df['Close'] > MA)).fillna(False)
    strong_trend = ((ER > trend_level) & (ER < max_level)).fillna(False)

    # Initialize the signal column
    signals['ERTrend_signals'] = 0

    # Apply buy signals
    signals.loc[buy_flag & strong_trend, 'ERTrend_signals'] = 1
    
    # Apply sell-to-close signal
    sell_close_condition = (stock_df['Close'].shift(-1) < MA) & (stock_df['Close'].shift(-2) > MA)
    sell_close_condition = sell_close_condition.fillna(False)
    signals.loc[sell_close_condition, 'ERTrend_signals'] = 2

    # Apply sell signals
    signals.loc[~buy_flag & strong_trend, 'ERTrend_signals'] = 3

    # Apply buy-to-close signal
    buy_close_condition = (stock_df['Close'].shift(-1) > MA) & (stock_df['Close'].shift(-2) < MA)
    buy_close_condition = buy_close_condition.fillna(False)
    signals.loc[buy_close_condition, 'ERTrend_signals'] = 4

    return signals

# Functions from firsthourbreakout.py
import pandas as pd
import pytz

# First Hour Breakout Strategy
def FirstHourBreakout(data):
    # Ensure data is indexed by datetime
    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize(pytz.utc)
    data.index = data.index.tz_convert(pytz.timezone('US/Eastern'))
    
    signals = pd.DataFrame(index=data.index)
    signals['FirstHourBreakout_signals'] = 0  # Default signal is 0
    
    # Define times
    market_open = pd.Timestamp('09:30', tz='US/Eastern').time()
    first_hour_end = pd.Timestamp('10:30', tz='US/Eastern').time()
    market_close = pd.Timestamp('16:15', tz='US/Eastern').time()

    # Group by normalized date to avoid filtering issues
    grouped = data.groupby(data.index.normalize())

    for day, day_data in grouped:
        if day_data.empty:
            continue
        
        # Calculate high and low of first trading hour
        first_hour_data = day_data.between_time(market_open, first_hour_end)
        if first_hour_data.empty:
            continue
        
        first_hour_high = first_hour_data['High'].max()
        first_hour_low = first_hour_data['Low'].min()

        # Generate signals within the day
        breakout_signals = day_data.apply(
            lambda row: 1 if row['High'] > first_hour_high else (2 if row['Low'] < first_hour_low else 0),
            axis=1
        )

        # Assign signals to the original DataFrame
        signals.loc[day_data.index, 'FirstHourBreakout_signals'] = breakout_signals

        # Close positions at the end of the day
        end_of_day_index = day_data.index[day_data.index.time == market_close]
        signals.loc[end_of_day_index, 'FirstHourBreakout_signals'] = 0

    return signals


# Functions from fourdaybreakoutle.py
import pandas as pd
import numpy as np
from ta import trend

def four_day_breakout_le_signals(stock_df, average_length=20, pattern_length=4, breakout_amount=0.50):
    # Calculate the Simple Moving Average
    sma = trend.SMAIndicator(stock_df['Close'], window=average_length)
    stock_df['SMA'] = sma.sma_indicator()
    
    # Identify all bullish candles
    stock_df['Bullish'] = np.where(stock_df['Close'] > stock_df['Open'], 1, 0)
    
    # Check if the last four candles are all bullish
    stock_df['Bullish_Count'] = stock_df['Bullish'].rolling(window=pattern_length).sum()
    
    # Identify the high of the highest candle in the pattern
    max_pattern_high = stock_df['High'].rolling(window=pattern_length).max().shift()
    
    # Create an empty signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0
    
    # Create a signal for when the close price is greater than the SMA, 
    # the last four candles are bullish, and the close price is greater 
    # than the high of the highest candle in the pattern by at least the breakout amount
    signals['Signal'] = np.where((stock_df['Close'] > stock_df['SMA']) &
                                 (stock_df['Bullish_Count'] == pattern_length) &
                                 (stock_df['Close'] > (max_pattern_high + breakout_amount)), 1, 0)
    
    # Create a column for simplified, visual-friendly signals
    signals['four_day_breakout_le_signals'] = 0
    signals.loc[signals['Signal'] == 1, 'four_day_breakout_le_signals'] = 1
    
    # Remove all the other columns
    signals.drop(['Signal'], axis=1, inplace=True)

    return signals


# Functions from gandalfprojectresearchsystem.py
import pandas as pd
import numpy as np

# GandalfProjectResearchSystem Strategy
def gandalf_signals(df, exit_length=10, gain_exit_length=20):  
    signals = pd.DataFrame(index=df.index)
    
    # Calculating ohlc4
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # Calculating median price
    df['median_price'] = (df['High'] + df['Low']) / 2
    # Calculating mid-body price
    df['mid_body'] = (df['Open'] + df['Close']) / 2
    
    # Buy signal
    signals['gandalf_buy_signals'] = 0
    signals.loc[((df['ohlc4'].shift(1) < df['median_price'].shift(1)) &
                 (df['median_price'].shift(2) <= df['ohlc4'].shift(1)) &
                 (df['median_price'].shift(2) <= df['ohlc4'].shift(3))) |
                ((df['ohlc4'].shift(1) < df['median_price'].shift(3)) &
                 (df['mid_body'] < df['median_price'].shift(2)) &
                 (df['mid_body'].shift(1) < df['mid_body'].shift(2))), 'gandalf_buy_signals'] = 1

    # Sell signal
    signals['gandalf_sell_signal'] = 0
    buy_indices = signals.index[signals['gandalf_signal'] == 1]
    
    for buy_index in buy_indices:
        buy_timestamp = signals.index[signals.index == buy_index]
        
        if not buy_timestamp.empty:
            buy_timestamp = buy_timestamp[0]  # Get the timestamp
            
            # Ensure buy_timestamp exists in df before accessing values
            if buy_timestamp in df.index:
                buy_open_price = df.at[buy_timestamp, 'Open']  # Scalar value access
                
                sell_condition = ((signals.index >= buy_timestamp + pd.Timedelta(days=exit_length)) |
                                  ((signals.index >= buy_timestamp + pd.Timedelta(days=gain_exit_length)) & 
                                   (df['Close'] > buy_open_price)) |
                                  ((df['Close'] < buy_open_price) &
                                   (((df['ohlc4'].shift(-1) < df['mid_body'].shift(-1)) &
                                     (df['median_price'].shift(-2) == df['mid_body'].shift(-3)) &
                                     (df['mid_body'].shift(-1) <= df['mid_body'].shift(-4))) |
                                    ((df['ohlc4'].shift(-2) < df['mid_body']) &
                                     (df['median_price'].shift(-4) < df['ohlc4'].shift(-3)) &
                                     (df['mid_body'].shift(-1) < df['ohlc4'].shift(-1))))))
                
                signals.loc[sell_condition, 'gandalf_signal'] = 2
    
    signals.fillna(0, inplace=True)
    return signals

# Functions from gapdownse.py
import pandas as pd
import numpy as np

# GapDownSE Strategy
def gap_down_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df["High"]
    signals['prev_low'] = stock_df["Low"].shift(1)
    signals['gap_down_se_signals'] = np.where(signals['high'] < signals['prev_low'], 'short', 'neutral')
    signals.drop(['high', 'prev_low'], axis=1, inplace=True)
    return signals


# Functions from gapmomentumsystem.py
import pandas as pd
import numpy as np

def gap_momentum_signals(data_df, length=14, signal_length=9, full_range=False):
    signals = pd.DataFrame(index=data_df.index)
    # calculate the gap prices
    gaps = data_df['Open'] - data_df['Close'].shift()
    # calculate the signal line 
    signals['gap_avg'] = gaps.rolling(window=length).mean() 
    # Calculate the signal based on the gap average
    signals['gap_momentum_signals'] = 'neutral'
    signals.loc[(signals['gap_avg'] > signals['gap_avg'].shift()), 'gap_momentum_signals'] = 'buy'
    signals.loc[(signals['gap_avg'] < signals['gap_avg'].shift()), 'gap_momentum_signals'] = 'sell'
    signals.drop(['gap_avg'], axis=1, inplace=True)
    return signals

# Functions from gapreversalle.py
import pandas as pd
import numpy as np

def gap_reversal_signals(stock_df, gap=0.10, offset=0.50):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the gap from the previous day's low
    stock_df['PrevLow'] = stock_df['Low'].shift(1)
    stock_df['Gap'] = (stock_df['Open'] - stock_df['PrevLow']) / stock_df['PrevLow']

    # Calculate the offset from the gap
    stock_df['Offset'] = (stock_df['High'] - stock_df['Open'])
    
    # Generate signals based on condition
    signals['gap_reversal_signals'] = 'Hold'
    signals.loc[(stock_df['Gap'] > gap) & (stock_df['Offset'] > offset), 'gap_reversal_signals'] = 'Buy'
    
    return signals


# Functions from gapuple.py
import pandas as pd

def gap_up_le_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['gap_up_le_signals'] = 'neutral'
    
    # Identify rows where the current Low is higher than the previous High
    gap_up = (stock_df['Low'] > stock_df['High'].shift(1))
    
    # Shift the signal to the next row
    shifted_gap_up = gap_up.shift(-1).fillna(False)  # Fill NaN with False to avoid indexer errors
    
    # Generate Long Entry signal for the next bar
    signals.loc[shifted_gap_up, 'gap_up_le_signals'] = 'long'
    
    return signals

# Functions from goldencrossbreakouts.py
import pandas as pd
import numpy as np
from ta import trend

# GoldenCrossBreakouts strategy
def golden_cross_signals(stock_df, fast_length=50, slow_length=200, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)

    if average_type == 'simple':
        fast_ma = trend.SMAIndicator(stock_df['Close'], window=fast_length).sma_indicator()
        slow_ma = trend.SMAIndicator(stock_df['Close'], window=slow_length).sma_indicator()
    elif average_type == 'exponential':
        fast_ma = trend.EMAIndicator(stock_df['Close'], window=fast_length).ema_indicator()
        slow_ma = trend.EMAIndicator(stock_df['Close'], window=slow_length).ema_indicator()
    else:
        raise ValueError("Invalid moving average type. Choose 'simple' or 'exponential'")

    signals['FastMA'] = fast_ma
    signals['SlowMA'] = slow_ma
    signals['golden_cross_signal'] = 'neutral'
    signals.loc[signals['FastMA'] > signals['SlowMA'], 'golden_cross_signal'] = 'long'
    signals.loc[signals['FastMA'] < signals['SlowMA'], 'golden_cross_signal'] = 'short'
    signals.drop(columns=['FastMA', 'SlowMA'], inplace=True)

    return signals


# Functions from goldentrianglele.py
import pandas as pd
import numpy as np
from ta import trend

def golden_triangle_signals(stock_df, average_length=50, confirm_length=20, volume_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    sma_long = trend.SMAIndicator(stock_df['Close'], average_length)
    sma_short = trend.SMAIndicator(stock_df['Close'], confirm_length)
    
    # Identify initial uptrend condition
    signals['uptrend'] = stock_df['Close'] > sma_long.sma_indicator()
    
    # Identify pivot points as local maxima
    signals['pivot'] = ((stock_df['Close'] > stock_df['Close'].shift()) &
                         (stock_df['Close'] > stock_df['Close'].shift(-1)))
    
    # Identify price drop condition
    signals['price_drop'] = stock_df['Close'] < sma_long.sma_indicator()
    
    # Define initial triangle setup condition
    signals['triangle_setup'] = np.where((signals['uptrend'] & 
                                          signals['pivot'] & 
                                          signals['price_drop']).shift().fillna(0), 'yes', 'no')
    
    # Price and volume confirmation
    signals['price_confirm'] = stock_df['Close'] > sma_short.sma_indicator()
    signals['volume_confirm'] = stock_df['Volume'] > stock_df['Volume'].rolling(volume_length).max()
    signals['triangle_confirm'] = np.where((signals['price_confirm'] & 
                                            signals['volume_confirm']).shift().fillna(0), 'yes', 'no')
    
    # Add a simulated Buy order when the triangle is confirmed
    signals['golden_triangle_le'] = np.where((signals['triangle_setup'] == 'yes') & 
                                             (signals['triangle_confirm'] == 'yes'), 'buy', 'wait')

    # Remove intermediate signals used for calculations
    signals = signals.drop(['uptrend', 'pivot', 'price_drop'], axis=1)

    return signals



# Functions from hacoltstrat.py
import pandas as pd
import numpy as np
import talib 

def hacolt_signals(data, tema_length=14, ema_length=9, candle_size_factor=0.7):
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index)
    
    # Calculate EMA
    ema = talib.EMA(data['Close'], timeperiod=ema_length)
    
    # Calculate TEMA
    tema = talib.T3(data['Close'], timeperiod=tema_length, vfactor=candle_size_factor)
    
    # Calculate HACOLT using EMA and TEMA
    hacolt = (ema / tema) * 100
    
    # Create a column for HACOLT values
    signals['hacolt'] = hacolt
    
    # Create empty signal column
    signals['hacolt_signal'] = 'neutral'
    
    # Create long entry, short entry, and long exit signals
    signals.loc[hacolt == 100, 'hacolt_signal'] = 'long_entry'
    signals.loc[hacolt == 0, 'hacolt_signal'] = 'short_entry'
    signals.loc[(hacolt != 100) & (signals['hacolt_signal'].shift() == 'long_entry'), 'hacolt_signal'] = 'long_exit'
    
    return signals




# Functions from halloween.py
import pandas as pd
from ta import trend

def halloween_strategy(data: pd.DataFrame, sma_length: int = 30):
    signals = pd.DataFrame(index=data.index)
    
    
    # Create SMA Indicator
    sma = trend.SMAIndicator(data["Close"], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Create a signal column and initialize it to Hold.
    signals['signal'] = 'Hold'
    
    # Generate Long Entry signal
    signals.loc[(signals.index.month == 10) & (signals.index.day == 1) & (data['Close'] > signals['SMA']), 'signal'] = 'Buy'
    
    # Generate Long Exit Signal
    signals.loc[(signals.index.month == 5) & (signals.index.day == 1), 'signal'] = 'Sell'
    signals.drop(['SMA'], axis=1, inplace=True)
    return signals


# Functions from ift_stoch.py
import pandas as pd
import numpy as np
import talib
from ta import trend

# IFT_Stoch Strategy using 'talib'
def ift_stoch_signals(df, length=14, slowing_length=3, over_bought=60, over_sold=30, sma_length=165):
    signals = pd.DataFrame(index=df.index)
    
    # Calculate Stochastic Oscillator using 'talib'
    stoch_k, stoch_d = talib.STOCH(df['High'], df['Low'], df['Close'], 
                                   fastk_period=length, slowk_period=slowing_length, slowd_period=slowing_length)

    signals['IFTStoch'] = 0.5 * ((stoch_k - 50) * 2) ** 2
    
    # Calculate SMA
    sma = trend.SMAIndicator(df['Close'], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Initialize all signals with neutral
    signals['ift_stoch_signals'] = 'neutral'
    
    # Generate signals according to given conditions
    signals.loc[(signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 'ift_stoch_signals'] = 'buy_to_open'
    signals.loc[(signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought) & (df['Close'] < signals['SMA']), 'ift_stoch_signals'] = 'sell_to_open'
    signals.loc[(signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought), 'ift_stoch_signals'] = 'sell_to_close'
    signals.loc[(signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 'ift_stoch_signals'] = 'buy_to_close'
    
    signals.drop(['IFTStoch', 'SMA'], axis=1, inplace=True)
    return signals



# Functions from insidebarle.py
import pandas as pd

def inside_bar_le(df):
    # Create a copy of the DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create a new column for the Long Entry signals
    signals['inside_bar_le_signals'] = False

    # Calculate the Inside Bar condition
    signals['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    
    # Generate the Long Entry signal condition
    signals.loc[signals['Inside_Bar'] & (df['Close'] > df['Open']), 'inside_bar_le_signals'] = True
    signals.drop(['Inside_Bar'], axis=1, inplace=True)
    return signals['inside_bar_le_signals']


# Functions from insidebarse.py
import pandas as pd
import numpy as np

# InsideBarSE Strategy
def inside_bar_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Identify the inside bars and close price is lower than open
    signals['inside_bar'] = np.where((stock_df['High'] < stock_df['High'].shift(1)) &
                                     (stock_df['Low'] > stock_df['Low'].shift(1)) &
                                     (stock_df['Close'] < stock_df['Open']), 1, 0)

    # Generate signals
    signals['inside_bar_se_signal'] = 'neutral'
    signals.loc[(signals['inside_bar'].shift(1) == 1), 'inside_bar_se_signal'] = 'Short Entry'
    signals.drop(['inside_bar'], axis=1, inplace=True)
    return signals


# Functions from keyrevle.py
import pandas as pd
import numpy as np

def key_rev_le_signals(stock_df, length=5):
    """
    Generate KeyRevLE strategy signals.
    :param stock_df: OHLCV dataset.
    :param length: The number of preceding bars whose Low prices are compared to the current Low.
    :return: DataFrame with 'key_rev_le_signal' column.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['key_rev_le_signal'] = 0

    for i in range(length, len(stock_df) - 1):  # Avoid index error at the end
        if stock_df['Low'].iloc[i] < stock_df['Low'].iloc[i-length:i].min() and \
           stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i-1]:
            signals.loc[signals.index[i+1], 'key_rev_le_signal'] = 1  # Fix using `.loc[]`
            
    return signals



# Functions from keyrevlx.py
import pandas as pd
import numpy as np

def key_rev_lx_signals(stock_df, length=5):
    """
    Generates KeyRevLX strategy signals.

    Parameters:
      stock_df (pd.DataFrame): stock dataset with columns: 'Open', 'High', 'Low', 'Close', 'Volume'
      length: The number of preceding bars whose High prices are compared to the current High.

    Returns:
      pd.DataFrame: signals with long exit   
    """

    # Create DataFrame for signals.
    signals = pd.DataFrame(index=stock_df.index)
    signals['key_rev_lx_signals'] = 0

    # Check the condition for each row
    for i in range(length, len(stock_df)):
        if stock_df['High'].iloc[i] > stock_df['High'].iloc[i-length:i].max() and \
           stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i-1]:
            signals.iloc[i, signals.columns.get_loc('key_rev_lx_signals')] = -1  # Use .iloc[] for safe modification

    return signals



# Functions from macdstrat.py
import pandas as pd
import numpy as np
from ta import trend
def macd_signals(df, fast_length=12, slow_length=26, macd_length=9):
    # Create a signals DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create MACD indicator
    macd = trend.MACD(df['Close'], window_slow=slow_length, window_fast=fast_length, window_sign=macd_length)

    # Generate MACD line and signal line
    signals['MACD_line'] = macd.macd()
    signals['MACD_signal'] = macd.macd_signal()
    
    # Create a column for the macd strategy signal
    signals['macd_strat'] = 0.0

    # Create signals: When the MACD line crosses the signal line upward, buy the stock
    signals['macd_strat'][(signals['MACD_line'] > signals['MACD_signal']) & (signals['MACD_line'].shift(1) < signals['MACD_signal'].shift(1))] = 1.0
    
    # When the MACD line crosses the signal line downward, sell the stock
    signals['macd_strat'][(signals['MACD_line'] < signals['MACD_signal']) & (signals['MACD_line'].shift(1) > signals['MACD_signal'].shift(1))] = -1.0
    signals.drop(['MACD_line','MACD_signal'], axis=1, inplace=True)
    return signals

# Functions from meanreversionswingle.py
import pandas as pd

# Function to detect uptrend based on moving averages
def detect_uptrend(close_prices, min_length=20, min_range_for_uptrend=5.0):
    rolling_min = close_prices.rolling(window=min_length).min()
    rolling_max = close_prices.rolling(window=min_length).max()
    return (rolling_max - rolling_min) > min_range_for_uptrend

def detect_pullback(close_prices, tolerance=1.0):
    close_prices = pd.to_numeric(close_prices, errors='coerce').ffill()  # Ensure numeric data
    rolling_high = close_prices.rolling(window=10).max()
    return (rolling_high - close_prices) > tolerance

def detect_up_move(close_prices, pullback_signal, min_up_move=0.5):
    close_prices = pd.to_numeric(close_prices, errors='coerce').ffill()  # Ensure numeric
    price_diff = close_prices.diff()
    return (price_diff > min_up_move) & pullback_signal

def limit_pattern_length(signals, max_length=400):
    if 'mean_reversion_swing_le' not in signals:
        return signals  # Avoid errors if column is missing

    signals['pattern_length'] = signals['mean_reversion_swing_le'].groupby(
        (signals['mean_reversion_swing_le'] != signals['mean_reversion_swing_le'].shift()).cumsum()
    ).cumcount() + 1

    signals.loc[signals['pattern_length'] > max_length, 'mean_reversion_swing_le'] = 'neutral'
    return signals.drop(columns=['pattern_length'])

def mean_reversion_swing_le(stock_df, min_length=20, max_length=400, min_range_for_uptrend=5.0,
                            min_up_move=0.5, tolerance=1.0):
    signals = pd.DataFrame(index=stock_df.index)

    # Ensure Close column is numeric
    stock_df['Close'] = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()

    # Compute trend-based conditions
    signals['uptrend'] = detect_uptrend(stock_df['Close'], min_length, min_range_for_uptrend)

    # Ensure detect_pullback() receives a numeric Close series
    signals['pullback'] = detect_pullback(stock_df['Close'], tolerance)

    # Ensure detect_up_move() receives valid inputs
    signals['up_move'] = detect_up_move(stock_df['Close'], signals['pullback'], min_up_move)

    # Generate trade signals
    signals['mean_reversion_swing_le'] = 'neutral'
    signals.loc[signals['uptrend'] & signals['pullback'] & signals['up_move'], 'mean_reversion_swing_le'] = 'long'

    # Ensure the pattern length does not exceed max_length
    signals = limit_pattern_length(signals, max_length)

    # Drop intermediate calculation columns
    signals = signals.drop(columns=['uptrend', 'pullback', 'up_move'])

    return signals


# Functions from middlehighlowmastrat.py
import pandas as pd
import numpy as np

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
    signals['MHLMA_signals'] = 0
    signals.loc[signals['MA'] > signals['MHL_MA'], 'MHLMA_signals'] = 1
    signals.loc[signals['MA'] < signals['MHL_MA'], 'MHLMA_signals'] = -1

    # Drop intermediate columns
    signals = signals.drop(['MA'], axis=1)

    return signals


# Functions from momentumle.py
import pandas as pd
import numpy as np
from ta import momentum

# MomentumLE Strategy
def momentumle_signals(stock_df, length=12, price_scale=100):
    # Initialize the signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Momentum
    mom = momentum.roc(stock_df['Close'], window=length)
    
    # Create 'momentumle_signals' column and set to 'neutral'
    signals['momentumle_signals'] = 'neutral'
    
    # Create a Long Entry signal when Momentum becomes a positive value and continues rising
    signals.loc[(mom > 0) & (mom.shift(1) <= 0), 'momentumle_signals'] = 'long'
    
    # Calculate signal price level
    signals['signal_price_level'] = stock_df['High'] + (1 / price_scale)
    
    # If the next Open price is greater than the current High plus one point, 
    # it is considered a price level to generate the signal at
    signals.loc[stock_df['Open'].shift(-1) > signals['signal_price_level'], 'momentumle_signals'] = 'long'
    signals = signals.drop(['signal_price_level'], axis=1)

    return signals


# Functions from movavgstrat.py
import pandas as pd

def moving_average_strategy(df, window=15, average_type='simple', mode='trend Following'):

    # compute moving average
    if average_type == 'simple':
        df['moving_avg'] = df['Close'].rolling(window=window).mean()
    elif average_type == 'exponential':
        df['moving_avg'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # create signals based on the mode
    df['moving_average_strategy_siganls'] = None
    if mode == 'trend Following':
        df.loc[df['Close'] > df['moving_avg'], 'moving_average_strategy_siganls'] = 'Buy' 
        df.loc[df['Close'] < df['moving_avg'], 'moving_average_strategy_siganls'] = 'Sell' 
    elif mode =='reversal':
        df.loc[df['Close'] > df['moving_avg'], 'moving_average_strategy_siganls'] = 'Sell'
        df.loc[df['Close'] < df['moving_avg'], 'moving_average_strategy_siganls'] = 'Buy' 
    
    return df['moving_average_strategy_siganls']


# Functions from movavgtwolinesstrat.py
# Import Necessary Libraries
import pandas as pd
import talib

# Moving Average Two Lines strategy
def mov_avg_two_lines_signals(stock_df, fast_length=5, slow_length=20, average_type='EMA'):
    signals = pd.DataFrame(index=stock_df.index)
    price = stock_df['Close']

    # Calculate Fast Moving Average
    if average_type == 'SMA':
        fastMA = talib.SMA(price, timeperiod=fast_length)
    elif average_type == 'WMA':
        fastMA = talib.WMA(price, timeperiod=fast_length)
    elif average_type == 'Wilder':
        fastMA = talib.WILDERS(price, timeperiod=fast_length)
    elif average_type == 'Hull':
        fastMA = talib.WMA(price, timeperiod=fast_length) # Hull is not directly supported by TA-Lib
    else:
        fastMA = talib.EMA(price, timeperiod=fast_length)

    # Calculate Slow Moving Average
    if average_type == 'SMA':
        slowMA = talib.SMA(price, timeperiod=slow_length)
    elif average_type == 'WMA':
        slowMA = talib.WMA(price, timeperiod=slow_length)
    elif average_type == 'Wilder':
        slowMA = talib.WILDERS(price, timeperiod=slow_length)
    elif average_type == 'Hull':
        slowMA = talib.WMA(price, timeperiod=slow_length) # Hull is not directly supported by TA-Lib
    else:
        slowMA = talib.EMA(price, timeperiod=slow_length)

    signals['FastMA'] = fastMA
    signals['SlowMA'] = slowMA
    signals['mov_avg_two_lines_signals'] = 'neutral'
    signals.loc[(signals['FastMA'] > signals['SlowMA']) & (signals['FastMA'].shift(1) <= signals['SlowMA'].shift(1)), 'mov_avg_two_lines_signals'] = 'long'
    signals.loc[(signals['FastMA'] < signals['SlowMA']) & (signals['FastMA'].shift(1) >= signals['SlowMA'].shift(1)), 'mov_avg_two_lines_signals'] = 'short'

    signals.drop(['FastMA', 'SlowMA'], axis=1, inplace=True)
    return signals


# Functions from onsettrend.py
import pandas as pd

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
    signals['onset_trend_detector_signals'] = 'neutral'
    signals.loc[(signals['oscillator1'] > 0) & (signals['oscillator1'].shift(1) <= 0), 'onset_trend_detector_signals'] = 'long'
    signals.loc[(signals['oscillator2'] < 0) & (signals['oscillator2'].shift(1) >= 0), 'onset_trend_detector_signals'] = 'short'
    
    signals.drop(['oscillator1', 'oscillator2'], axis=1, inplace=True)
    
    return signals


# Functions from pmostrat.py
import pandas as pd
import numpy as np

# Price Momentum Oscillator (PMO) Strategy
def pmo_signals(stock_df, length1=20, length2=10, signal_length=10):
    # Calculate the one-bar rate of change
    roc = stock_df['Close'].pct_change()
    
    # Smooth the rate of change using two exponential moving averages
    pmo_line = roc.ewm(span=length1, adjust=False).mean().ewm(span=length2, adjust=False).mean()
    
    # Create the signal line, which is an EMA of the PMO line
    pmo_signal = pmo_line.ewm(span=signal_length, adjust=False).mean()
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    signals['pmo_line'] = pmo_line
    signals['pmo_signals'] = pmo_signal

    # Generate trading signals based on crossover of PMO line and PMO signal
    signals['buy'] = np.where((signals['pmo_line'] > signals['pmo_signals']) & (signals['pmo_line'].shift(1) <= signals['pmo_signals'].shift(1)), 1, 0)
    signals['sell'] = np.where((signals['pmo_line'] < signals['pmo_signals']) & (signals['pmo_line'].shift(1) >= signals['pmo_signals'].shift(1)), -1, 0)
    
    # 'pmo_signals' column contains buy(1)/sell(-1)/neutral(0) signals
    signals['pmo_signals'] = signals['buy'] + signals['sell']
    
    signals.drop(['pmo_line', 'buy', 'sell'], axis=1, inplace=True)

    return signals


# Functions from priceswing.py

from ta import momentum, volatility
import numpy as np
import pandas as pd

def price_swing_signals(stock_df, swing_type="RSI", length=20, exit_length=20, deviations=2, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=stock_df['Close'], window=length, window_dev=deviations)
    stock_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    stock_df['bb_bbh'] = indicator_bb.bollinger_hband()
    stock_df['bb_bbl'] = indicator_bb.bollinger_lband()

    if swing_type == "bollinger":
        # Use Bollinger Bands crossover swing type
        signals['price_swing_signals'] = np.where(stock_df['Close'] > stock_df['bb_bbh'], 'down', 'neutral')
        signals['price_swing_signals'] = np.where(stock_df['Close'] < stock_df['bb_bbl'], 'up', signals['price_swing_signals'])

    elif swing_type == "RSI":
        # Use RSI crossover swing type
        rsi = momentum.RSIIndicator(close=stock_df['Close'], window=length)
        signals['rsi'] = rsi.rsi()
        signals['price_swing_signals'] = np.where(signals['rsi'] > overbought, 'down', 'neutral')
        signals['price_swing_signals'] = np.where(signals['rsi'] < oversold, 'up', signals['price_swing_signals'])
        signals.drop(['rsi'], axis=1, inplace=True)

    elif swing_type == "RSI_HighLow":
        # Use RSI + Higher Low/Lower High swing type
        # Insert your RSI + Higher Low/Lower High swing type strategy here. 
        pass

    # Add simulated exit orders after exit_length
    signals['exit'] = signals['price_swing_signals'].shift(exit_length)
        
    signals.drop(['exit'], axis=1, inplace=True)

    return signals



# Functions from pricezoneoscillatorle.py
import pandas as pd
import numpy as np
from ta import trend, volatility

# Price Zone Oscillator (PZO) Strategy
def pzo_signals(stock_df, length=14, ema_length=60):
    signals = pd.DataFrame(index=stock_df.index)
    pzo = ((stock_df['Close'] - stock_df['Close'].rolling(window=length).mean()) / stock_df['Close'].rolling(window=length).std())*100
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=length).adx()
    ema = stock_df['Close'].ewm(span=ema_length).mean()

    signals['PZO'] = pzo
    signals['ADX'] = adx
    signals['EMA'] = ema
    signals['pzo_le_signals'] = 'neutral'

    # ADX > 18, price > EMA, and PZO cross "-40" level or surpass "+15" level from below
    signals.loc[(signals['ADX'] > 18) & (stock_df['Close'] > signals['EMA']) & (
        (signals['PZO'].shift(1) < -40) & (signals['PZO'] > -40) |
        ((signals['PZO'].shift(1) < 0) & (signals['PZO'] > 0) & (signals['PZO'] > 15))), 'pzo_le_signals'] = 'long'
    
    # ADX < 18, and PZO cross "-40" or "+15" level from below
    signals.loc[(signals['ADX'] <= 18) & (
        (signals['PZO'].shift(1) < -40) & (signals['PZO'] > -40) |
        (signals['PZO'].shift(1) < 15) & (signals['PZO'] > 15)), 'pzo_le_signals'] = 'long'

    signals.drop(['PZO','ADX','EMA'], axis=1, inplace=True)
    return signals


# Functions from pricezoneoscillatorlx.py
import pandas as pd
import numpy as np
from ta import trend, volatility

# PriceZoneOscillatorLX Strategy
def pzo_signals(df, length=14, ema_length=60):
    # Calculate EMA
    ema = trend.EMAIndicator(df['Close'], ema_length).ema_indicator() 
    
    # Calculate ADX
    adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'], length).adx()
    
    # Calculate Bollinger Bands
    bb = volatility.BollingerBands(df['Close'], window=length)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    
    # Calculate PZO
    pzo = (df['Close'] - ((upper + lower) / 2)) / ((upper - lower) / 2) * 100
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=df.index)
    signals['pzo_lx_signals'] = pzo
    signals['pzo_lx_trend'] = 'neutral'
    
    # Set conditions for Long Exit signals
    conditions = [
        (adx > 18) & (pzo > 60) & (pzo < pzo.shift(1)),  # PZO above +60 and going down in trending 
        (adx > 18) & (df['Close'] < ema) & (pzo < 0),    # PZO negative and price below EMA in trending 
        (adx < 18) & (pzo.shift(1) > 40) & (pzo < 0) & (df['Close'] < ema),     # PZO below zero with prior crossing +40 and price below EMA in non-trending
        (adx < 18) & (pzo.shift(1) < 15) & (pzo > -5) & (pzo < 40)    # PZO failed to rise above -40, instead fell below -5 in non-trending
    ]
    
    choices = ['long_exit'] * 4
    
    # Apply conditions and choices
    signals['pzo_lx_trend'] = np.select(conditions, choices, 'neutral')
    
    return signals


# Functions from pricezoneoscillatorse.py
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



# Functions from pricezoneoscillatorsx.py
#%%
import pandas as pd
from ta import trend
def calculate_PZO(df, length=14):
    EMA_high = df['High'].ewm(span=length, adjust=False).mean()
    EMA_low = df['Low'].ewm(span=length, adjust=False).mean()
    EMA_close = df['Close'].ewm(span=length, adjust=False).mean()
    donchian_channel = EMA_high - EMA_low
    y = (EMA_close - ((EMA_high + EMA_low)/2)) / donchian_channel
    PZO = y*100
    return PZO

def PriceZoneOscillatorSX(df, length=14, ema_length=60):
    signals = pd.DataFrame(index=df.index)
    adx = trend.ADXIndicator(df['High'], df['Low'], df['Close'], length).adx()
    ema = df['Close'].ewm(span=ema_length, adjust=False).mean()
    PZO = calculate_PZO(df, length)

    signals['adx'] = adx
    signals['ema'] = ema
    signals['pzo'] = PZO

    # Initialize short_exit to zero
    signals['pzo_sx_signal'] = 0

    # Calculate short_exit conditions based on PZO, ADX and EMA values
    for i in range(2, signals.shape[0]):
        if (signals['adx'].iloc[i] > 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                (signals['pzo'].iloc[i] > 0 and df['Close'].iloc[i] > signals['ema'].iloc[i])):
                signals.loc[signals.index[i], 'pzo_sx_signal'] = 1
        elif (signals['adx'].iloc[i] < 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                ((signals['pzo'].iloc[i] > 0 or signals['pzo'].iloc[i-1] > -40) and df['Close'].iloc[i] > signals['ema'].iloc[i]) or
                (signals['pzo'].iloc[i] > 15 and signals['pzo'].iloc[i-1] <= -5 and signals['pzo'].iloc[i-2] > -40)):
                signals.loc[signals.index[i], 'pzo_sx_signal'] = 1
        
    signals.drop(['pzo', 'ema', 'adx'], axis=1, inplace=True)

    return signals


# Functions from profittargetlx.py
import pandas as pd
import numpy as np

def profit_target_lx_signals(stock_df, target=0.01, offset_type="percent"):
    signals = pd.DataFrame(index=stock_df.index)
    signals['profit_target_lx_signals'] = 'neutral'

    if offset_type=="percent":
        exit_price = stock_df['Close'] * (1 + target)
    elif offset_type=="tick":
        exit_price = stock_df['Close'] + (stock_df['Close'].diff() * target)
    elif offset_type=="value":
        exit_price = stock_df['Close'] + target
    else:
        return "Invalid offset type. Please use 'percent', 'tick' or 'value'."

    signals.loc[stock_df['Close'].shift(-1) >= exit_price, 'profit_target_lx_signals'] = 'long_exit'

    return signals

# Functions from profittargetsx.py
import pandas as pd
import numpy as np

def profit_target_SX(df, target=0.75, offset_type='value', tick_size=0.01):
    signals = pd.DataFrame(index=df.index)
    signals['profit_target_SX_signals'] = 0
    if offset_type == 'value':
        signals['profit_target_SX_signals'] = np.where(df['Close'].diff() <= -target, 'Short Exit', signals['profit_target_SX_signals'])
    elif offset_type == 'tick':
        signals['profit_target_SX_signals'] = np.where(df['Close'].diff() <= -(target * tick_size), 'Short Exit', signals['profit_target_SX_signals'])
    elif offset_type == 'percent':
        signals['profit_target_SX_signals'] = np.where(df['Close'].pct_change() <= -target/100, 'Short Exit', signals['profit_target_SX_signals'])  
    return signals

# Functions from r2trend.py
import pandas as pd
import numpy as np
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
    signals['r2trend_signals'] = 0

    # Strong Uptrend Condition
    cond1 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] > lr_crit_level) & \
            (df['Close'] > signals['ma'])

    signals.loc[cond1, 'r2trend_signals'] = 1

    # Strong Downtrend Condition
    cond2 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] < -lr_crit_level) & \
            (df['Close'] < signals['ma'])

    signals.loc[cond2, 'r2trend_signals'] = -1
    signals.drop(['slope', 'ma', 'RSquared_lag'], axis=1, inplace=True)

    return signals


# Functions from rateofchangewithbandsstrat.py
import pandas as pd
import numpy as np
from ta import momentum

def rocwb_signals(stock_df, roc_length=14, average_length=9, ema_length=12, num_rmss=2, average_type='simple'):

    # Create a DataFrame to hold signals
    signals = pd.DataFrame(index=stock_df.index)

    # Compute ROC
    roc = momentum.roc(stock_df['Close'], window=roc_length)
    signals['ROC'] = roc

    # Compute average ROC
    mov_avgs = {
        'simple': pd.Series.rolling,
        'exponential': pd.Series.ewm,
        # The following two would need the library `ta`
        # 'weighted': ta.volatility.bollinger_mavg
        # 'Wilder's': ta.trend.ema_indicator,
        # 'Hull': ta.trend.hma_indicator,
    }
    signals['AvgROC'] = mov_avgs[average_type](signals['ROC'], window=average_length).mean()

    # Compute RMS of ROC
    signals['RMS'] = np.sqrt(np.mean(np.square(signals['ROC'].diff().dropna())))

    # Compute bands
    signals['LowerBand'] = signals['AvgROC'] - num_rmss * signals['RMS']
    signals['UpperBand'] = signals['AvgROC'] + num_rmss * signals['RMS']

    # Compute EMA
    signals['EMA'] = stock_df['Close'].ewm(span=ema_length, adjust=False).mean()

    # Initialize the signal column as neutral
    signals['rocwb_signal'] = 'neutral'

    # Generate Buy signals (when Close is above EMA and ROC is above LowerBand)
    signals.loc[(stock_df['Close'] > signals['EMA']) & (signals['ROC'] > signals['LowerBand']), 'rocwb_signal'] = 'buy'

    # Generate Sell signals (when Close is below EMA and ROC is below UpperBand)
    signals.loc[(stock_df['Close'] < signals['EMA']) & (signals['ROC'] < signals['UpperBand']), 'rocwb_signal'] = 'sell'

    # Remove all auxiliary columns
    signals.drop(['ROC', 'AvgROC', 'RMS', 'LowerBand', 'UpperBand', 'EMA'], axis=1, inplace=True)

    # Return signals DataFrame
    return signals

# Functions from reverseemastrat.py
import numpy as np
import pandas as pd

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
    signals['trend_ema'] = reverse_ema(df['Close'], trend_length)
    signals['cycle_ema'] = reverse_ema(df['Close'], cycle_length)

    # Generate signals
    signals['buy_to_open'] = np.where((signals['cycle_ema'] > 0) & (signals['trend_ema'] > 0), 1, 0)
    signals['sell_to_close'] = np.where((signals['cycle_ema'] < 0) | (signals['trend_ema'] < 0), -1, 0)
    signals['sell_to_open'] = np.where((signals['cycle_ema'] < 0) & (signals['trend_ema'] < 0), -1, 0)
    signals['buy_to_close'] = np.where((signals['cycle_ema'] > 0) | (signals['trend_ema'] > 0), 1, 0)

    # If there's signal for both actions 'to open' and 'to close',
    # priority is given to 'close' actions
    signals['reverse_ema_strat_signals'] = np.where(
        (signals['buy_to_open'] + signals['sell_to_close']) == 0,
        signals['sell_to_open'],
        signals['buy_to_close']
    )

    return signals['reverse_ema_strat_signals']


# Functions from rsistrat.py
import pandas as pd
from ta import momentum

def rsi_signals(df, length=14, overbought=70, oversold=30, rsi_average_type='simple'):
    close_price = df['Close']

    if rsi_average_type == 'simple':
        rsi = momentum.RSIIndicator(close_price, window=length).rsi()
    elif rsi_average_type == 'exponential':
        rsi = close_price.ewm(span=length, min_periods=length - 1).mean()

    signals = pd.DataFrame(index=df.index)
    signals['RSI'] = rsi

    signals['RSI_strat_signal'] = 'Neutral'
    signals.loc[(signals['RSI'] > oversold) & (signals['RSI'].shift(1) <= oversold), 'RSI_strat_signal'] = 'Buy'
    signals.loc[(signals['RSI'] < overbought) & (signals['RSI'].shift(1) >= overbought), 'RSI_strat_signal'] = 'Sell'
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals



# Functions from rsitrend.py
from ta import momentum
import numpy as np
import pandas as pd
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
    signals['rsi_trend_signal'] = 'neutral'
    signals.loc[conditions_buy, 'rsi_trend_signal'] = 'buy'
    signals.loc[conditions_sell, 'rsi_trend_signal'] = 'sell'

    return signals

# Functions from simplemeanreversion.py
import pandas as pd
import numpy as np

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
    signals['simple_mean_reversion_signals'] = 'neutral'

    # Buy Signal: When Z-Score < -Entry Level and Faster SMA > Slower SMA
    signals.loc[(signals['zScore'] < -z_score_entry_level) & (signals['FasterSMA'] > signals['SlowerSMA']), 'simple_mean_reversion_signals'] = 'buy'

    # Sell Signal: When Z-Score > Entry Level and Faster SMA < Slower SMA
    signals.loc[(signals['zScore'] > z_score_entry_level) & (signals['FasterSMA'] < signals['SlowerSMA']), 'simple_mean_reversion_signals'] = 'sell'

    # Drop unnecessary columns
    signals = signals[['simple_mean_reversion_signals']]

    return signals


# Functions from simplerocstrat.py
import pandas as pd
import numpy as np
from ta import momentum

def roc_signals(ohlcv_df, signal_length=4, roc_length=2, rms_length=3, use_fm_demodulator=False):
    signals = pd.DataFrame(index=ohlcv_df.index)
    
    price = ohlcv_df['Close']
    roc_indicator = momentum.ROCIndicator(price, roc_length)
    momentum_roc = roc_indicator.roc()
    
    if use_fm_demodulator:
        fm_demodulator = momentum_roc.rolling(window=rms_length).apply(lambda x: np.sqrt(np.mean(x**2)))
        signal_var = fm_demodulator.rolling(window=signal_length).mean()
    else:
        signal_var = momentum_roc.rolling(window=signal_length).sum() / signal_length
    
    signals['ROC'] = signal_var
    signals['roc_signal'] = 'neutral'
    signals.loc[(signals['ROC'] > 0) & (signals['ROC'].shift(1) <= 0), 'roc_signal'] = 'long'
    signals.loc[(signals['ROC'] < 0) & (signals['ROC'].shift(1) >= 0), 'roc_signal'] = 'short'

    signals.drop(['ROC'], axis=1, inplace=True)
    return signals



# Functions from spectrumbarsle.py
import pandas as pd
import numpy as np

# SpectrumBarsLE Strategy
def spectrum_bars_le_signals(df, length=10):
    signals = pd.DataFrame(index=df.index)
    signals['close_shift'] = df['Close'].shift(length)
    
    # Define the SpectrumBarsLE conditions:
    # Close price is greater than that from a specified number of bars ago 
    signals['spectrum_bars_le_signals'] = np.where(df['Close'] > signals['close_shift'], 'long', 'neutral')

    # Drop unnecessary columns
    signals.drop(columns='close_shift', inplace=True)
    
    return signals

# Functions from stiffnessstrat.py
import pandas as pd
import numpy as np
from ta import trend, volatility

def get_stiffness_indicator(df, length=100, average_length=20, num_dev=2):
    sma = trend.SMAIndicator(df['Close'], window=int(average_length)).sma_indicator()
    bollinger = volatility.BollingerBands(df['Close'], window=int(average_length), window_dev=num_dev)
    upper_band = bollinger.bollinger_hband()
    condition = (df['Close'] > sma + upper_band)
    df['stiffness'] = condition.rolling(window=length).sum() / length * 100
    return df

def get_market_trend(df, market_index='Close', length=2):
    """
    Computes market trend using EMA of the given market index column.
    If market_index does not exist, it defaults to using 'Close' prices.
    """
    if market_index not in df.columns:
        print(f"Warning: {market_index} not found in dataset. Using 'Close' instead.")
        market_index = 'Close'

    df['ema'] = trend.EMAIndicator(df[market_index]).ema_indicator()
    uptrend = (df['ema'] > df['ema'].shift()) & (df['ema'].shift() > df['ema'].shift(2))
    df['uptrend'] = uptrend
    return df 

def StiffnessStrat(df, length=84, average_length=20, exit_length=84, num_dev=2, entry_stiffness_level=90, exit_stiffness_level=50, market_index='Close'):
    
    # Get stiffness and market trend
    df = get_stiffness_indicator(df, length=length, average_length=average_length, num_dev=num_dev)
    df = get_market_trend(df, market_index=market_index)
    
    # Entry and exit conditions
    entry = (df['uptrend'] & (df['stiffness'] > entry_stiffness_level)).shift()
    exit = ((df['stiffness'] < exit_stiffness_level) | ((df['stiffness'].shift().rolling(window=exit_length).count() == exit_length))).shift()
    
    df['Buy_Signal'] = np.where(entry, 'buy', 'neutral') 
    df['Sell_Signal'] = np.where(exit, 'sell', 'neutral') 

    # Combine into one signal column
    df['stiffness_strat_signal'] = 'neutral'
    df.loc[df['Buy_Signal'] == 'buy', 'stiffness_strat_signal'] = 'buy'
    df.loc[df['Sell_Signal'] == 'sell', 'stiffness_strat_signal'] = 'sell'

    return df[['stiffness_strat_signal']]


# Functions from stochastic.py
import pandas as pd
from ta import momentum

def stochastic_signals(stock_df, k=14, d=3, overbought=80, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    stoch = momentum.StochasticOscillator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=k, smooth_window=d)
    
    signals['stoch_k'] = stoch.stoch()
    signals['stoch_d'] = stoch.stoch_signal()
    
    signals['stochastic_strat_signals'] = 0
    # Create signal when 'stoch_k' crosses above 'stoch_d'
    signals.loc[signals['stoch_k'] > signals['stoch_d'], 'stochastic_strat_signals'] = 1
    # Create signal when 'stoch_k' crosses below 'stoch_d'
    signals.loc[signals['stoch_k'] < signals['stoch_d'], 'stochastic_strat_signals'] = -1
    
    # Create states of 'overbought' and 'oversold'
    signals['overbought'] = signals['stoch_k'] > overbought
    signals['oversold'] = signals['stoch_k'] < oversold
    signals.drop(['stoch_k', 'stoch_d', 'overbought','oversold'], axis=1, inplace=True)

    return signals



# Functions from stoplosslx.py

import pandas as pd
import numpy as np

def stop_loss_lx_signals(stock_df, offset_type="percent", stop=0.75):

    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_lx_signal'] = 'neutral'
            
    if offset_type.lower() == "value":
        stop_loss_price = stock_df['Close'].shift(1) - stop
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    if offset_type.lower() == "tick":
        stop_loss_price = stock_df['Close'].shift(1) - (stop * stock_df['TickSize'])
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    if offset_type.lower() == "percent":
        stop_loss_price = stock_df['Close'].shift(1) - (stock_df['Close'].shift(1) * stop/100)
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_lx_signal'] = 'exit'
        
    return signals


# Functions from stoplosssx.py
import pandas as pd
import numpy as np

def stop_loss_sx_signals(stock_df, offset_type='percent', stop=0.75):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_sx_signals'] = 0

    if offset_type.lower() == "value":
        signals.loc[(stock_df['Close'] - stock_df['Close'].shift() > stop), 'stop_loss_sx_signals'] = 1
    elif offset_type.lower() == "tick":
        tick_sizes = (stock_df['High'] - stock_df['Low']) / 2  # Assume the tick size is half the daily range
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / tick_sizes) > stop, 'stop_loss_sx_signals'] = 1
    elif offset_type.lower() == "percent":
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / stock_df['Close'].shift()) * 100 > stop, 'stop_loss_sx_signals'] = 1

    return signals



# Functions from svehatypcross.py
import numpy as np
import pandas as pd
import talib

def typical_price(df):
    return (df['High'] + df['Low'] + df['Close']) / 3

def HA(df):
    HA_Close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    HA_Open = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    HA_High = df[['High', 'Open', 'Close']].max(axis=1)
    HA_Low = df[['Low', 'Open', 'Close']].min(axis=1)
    return HA_Open, HA_High, HA_Low, HA_Close

def sve_ha_typ_cross_signals(df, typical_length=14, ha_length=14):
    signals = pd.DataFrame(index=df.index)
    
    tp = typical_price(df)
    tp_ema = talib.EMA(tp, timeperiod=typical_length)
    
    ha_open, ha_high, ha_low, ha_close = HA(df)
    ha_avg = (ha_open + ha_high + ha_low + ha_close) / 4
    ha_ema = talib.EMA(ha_avg, timeperiod=ha_length)

    # Initialize signal column with 'neutral'
    signals['sve_ha_typ_cross_signals'] = 'neutral'

    # Assign 'buy' and 'sell' signals
    signals.loc[(tp_ema > ha_ema) & (tp_ema.shift() < ha_ema.shift()), 'sve_ha_typ_cross_signals'] = 'buy'
    signals.loc[(tp_ema < ha_ema) & (tp_ema.shift() > ha_ema.shift()), 'sve_ha_typ_cross_signals'] = 'sell'

    return signals[['sve_ha_typ_cross_signals']]



# Functions from svesc.py
import pandas as pd
import numpy as np

def svesc_signals(stock_df, length=14, exit_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate hlc3
    signals['hlc3'] = (stock_df['High'] + stock_df['Low'] + stock_df['Close']) / 3
    
    # Calculate Heikin Ashi ohlc4
    ha_open = (stock_df['Open'].shift() + stock_df['Close'].shift()) / 2
    ha_high = stock_df[['High', 'Low', 'Close']].max(axis=1)
    ha_low = stock_df[['High', 'Low', 'Close']].min(axis=1)
    ha_close = (stock_df['Open'] + ha_high + ha_low + stock_df['Close']) / 4
    signals['ha_ohlc4'] = (ha_open + ha_high + ha_low + ha_close) / 4
    
    # Calculate averages
    signals['average_hlc3'] = signals['hlc3'].rolling(window=length).mean()
    signals['average_ha_ohlc4'] = signals['ha_ohlc4'].rolling(window=length).mean()
    signals['average_close'] = stock_df['Close'].rolling(window=exit_length).mean()
    
    # Conditions to simulate orders
    signals['long_entry'] = np.where(signals['average_hlc3'] > signals['average_ha_ohlc4'], 1, 0)
    signals['short_entry'] = np.where(signals['average_hlc3'] < signals['average_ha_ohlc4'], -1, 0)
    signals['long_exit'] = np.where((signals['short_entry'].shift() == -1) &(stock_df['Close'] > signals['average_close']) & (stock_df['Close'] > stock_df['Open']), 1, 0)
    signals['short_exit'] = np.where((signals['long_entry'].shift() == 1) & (stock_df['Close'] < signals['average_close']) & (stock_df['Close'] < stock_df['Open']), -1, 0)
    
    # Consolidate signals
    signals['orders'] = signals['long_entry'] + signals['short_entry'] + signals['long_exit'] + signals['short_exit']
    
    # Map signal to action
    signals['svesc_signals'] = 'hold'
    signals.loc[signals['orders'] > 0 , 'svesc_signals'] = 'long'
    signals.loc[signals['orders'] < 0 , 'svesc_signals'] = 'short'
    
    # Drop unnecessary columns
    signals.drop(columns=['hlc3', 'ha_ohlc4', 'average_hlc3', 'average_ha_ohlc4', 'average_close', 'long_entry', 'short_entry', 'long_exit', 'short_exit', 'orders'], inplace=True)
    
    return signals



# Functions from svezlrbpercbstrat.py
import pandas as pd
import numpy as np
from ta import momentum, volatility

def sve_zl_rb_perc_b_strat(stock_df, stddev_length=10, ema_length=10, num_dev=2, k_period=14, slowing_period=3):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Initialising Bollinger Bands
    bb = volatility.BollingerBands(close=stock_df['Close'], window=stddev_length, window_dev=num_dev)
    
    # Adding to signals df
    signals['percent_b'] = bb.bollinger_pband()
    
    # Initialising Stochastic Oscillator
    stoch = momentum.StochasticOscillator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=k_period, smooth_window=slowing_period)
    
    # Adding to signals df
    signals['stochastic'] = stoch.stoch_signal()
    
    # Creating the signals
    signals['sve_zl_rb_perc_b_strat'] = 'neutral'
    signals.loc[(signals['percent_b'] > signals['percent_b'].shift(1)) & (signals['stochastic'] > signals['stochastic'].shift(1)), 'sve_zl_rb_perc_b_strat'] = 'long'
    signals.loc[(signals['percent_b'] < signals['percent_b'].shift(1)) & (signals['stochastic'] < signals['stochastic'].shift(1)), 'sve_zl_rb_perc_b_strat'] = 'short'
    
    signals.drop(['percent_b', 'stochastic'], axis=1, inplace=True)

    return signals

# Functions from swingthree.py
import pandas as pd
import numpy as np
from ta import trend

# SwingThree Strategy
def swingthree_signals(stock_df, sma_length=14, ema_length=50, tick_sizes=5):
    signals = pd.DataFrame(index=stock_df.index)
    sma_high = trend.sma_indicator(stock_df['High'], sma_length)
    sma_low = trend.sma_indicator(stock_df['Low'], sma_length)
    ema_close = trend.ema_indicator(stock_df['Close'], ema_length)
    signals['sma_high'] = sma_high
    signals['sma_low'] = sma_low
    signals['ema_close'] = ema_close
    signals['swingthree_signals_long_entry'] = (stock_df['High'] > sma_high + tick_sizes) & (stock_df['Close'].shift(1) > ema_close)
    signals['long_exit'] = stock_df['Low'] <= sma_low
    signals['swingthree_signals_short_entry'] = (stock_df['Low'] < sma_low - tick_sizes) & (stock_df['Close'].shift(1) < ema_close)
    signals['short_exit'] = stock_df['High'] >= sma_high
    signals.drop(['sma_high', 'sma_low','short_exit','ema_close','long_exit'], axis=1, inplace=True)

    return signals



# Functions from threebarinsidebarle.py
import pandas as pd
import numpy as np

def three_bar_inside_bar_le(ohlcv_df):
    signals = pd.DataFrame(index=ohlcv_df.index)
    # Define the conditions for the three bar inside bar pattern
    condition1 = ohlcv_df['Close'].shift(2) < ohlcv_df['Close'].shift(1)
    condition2 = ohlcv_df['High'].shift(1) > ohlcv_df['High']
    condition3 = ohlcv_df['Low'].shift(1) < ohlcv_df['Low']
    condition4 = ohlcv_df['Close'].shift(1) < ohlcv_df['Close']
    
    # Aggregate all conditions
    conditions = condition1 & condition2 & condition3 & condition4

    # Generate the long entry signals
    signals['three_bar_inside_bar_le_signal'] = np.where(conditions.shift(-1), 'Long', 'Neutral')

    return signals



# Functions from threebarinsidebarse.py
import pandas as pd
import numpy as np

def three_bar_inside_bar_se(df):
    # Create a signal column initialized to 'neutral'
    signals = pd.DataFrame(index=df.index)
    signals['three_bar_inside_bar_se_signal'] = 'neutral'
    
    for i in range(3, len(df)):
        # The first bar's Close price is higher than that of the second bar 
        condition1 = df['Close'].iloc[i-3] > df['Close'].iloc[i-2]
        
        # The third bar's High price is less than that of the previous bar
        condition2 = df['High'].iloc[i-1] < df['High'].iloc[i-2]
        
        # The third bar's Low price is greater than that of the previous bar
        condition3 = df['Low'].iloc[i-1] > df['Low'].iloc[i-2]
        
        # The fourth bar's Close price is lower than that of the previous bar
        condition4 = df['Close'].iloc[i] < df['Close'].iloc[i-1]
        
        # If all conditions are met, generate a 'short' signal
        if condition1 and condition2 and condition3 and condition4:
            signals.loc[df.index[i], 'three_bar_inside_bar_se_signal'] = 'short'
    
    return signals




# Functions from trailingstoplx.py
import pandas as pd
import numpy as np

def trailing_stop_lx_signals(stock_df, trail_stop=1, offset_type='percent'):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Stop Price'] = np.nan
    signals['Long Exit'] = 0

    # Ensure rolling window is an integer and valid
    window_size = max(int(trail_stop), 1)

    if offset_type == 'percent':
        signals['Stop Price'] = stock_df['Close'].rolling(window_size).max() * (1 - trail_stop / 100)
    elif offset_type == 'value':
        signals['Stop Price'] = stock_df['Close'].rolling(window_size).max() - trail_stop
    else:  # Tick
        tick_size = 0.01  # Replace with actual tick size if known
        signals['Stop Price'] = stock_df['Close'].rolling(window_size).max() - (tick_size * trail_stop)

    # Generate exit signals
    signals.loc[stock_df['Low'] < signals['Stop Price'].shift(), 'Long Exit'] = 1

    return signals

# Functions from trailingstopsx.py
import pandas as pd
import numpy as np

def trailing_stop_sx_signals(stock_df, trail_stop=1.0, offset_type='percent'):
    signals = pd.DataFrame(index=stock_df.index)
    entry_price = stock_df['Close'].shift()  # Use the closing price as the entry price

    if offset_type == 'value':
        stop_price = entry_price + trail_stop
    elif offset_type == 'percent':
        stop_price = entry_price * (1 + trail_stop / 100)
    elif offset_type == 'tick':
        tick_size = 0.01  # for simplicity, assuming tick size is 0.01
        stop_price = entry_price + trail_stop * tick_size
    else:
        raise ValueError(f'Invalid offset_type "{offset_type}". Choose from "value", "percent", "tick"')

    signals['TrailingStop'] = np.maximum.accumulate(stop_price) # Accumulates the maximum stop price
    signals['trailing_stop_sx_signals'] = 0
    signals.loc[(stock_df['High'] > signals['TrailingStop']), 'trailing_stop_sx_signals'] = 1
    signals.drop(['TrailingStop'], axis=1, inplace=True)

    return signals



# Functions from trendfollowingstrat.py
import pandas as pd
import numpy as np
from ta import trend

def calculate_moving_avg(close, length=10, average_type='simple'):
    close = pd.to_numeric(close, errors='coerce').ffill()  # Ensure numeric data

    if average_type == 'simple':
        return close.rolling(window=length).mean()
    elif average_type == 'exponential':
        return close.ewm(span=length, adjust=False).mean()
    elif average_type == 'wilder':
        return trend.EMAIndicator(close=close, window=length, fillna=False).ema_indicator()
    elif average_type == 'hull':
        return trend.WMAIndicator(close=close, window=length, fillna=False).wma_indicator()
    else: 
        return None

def trend_following_signals(stock_df, length=10, average_type='simple', entry_percent=3.0, exit_percent=4.0):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Ensure Close column is numeric
    close = pd.to_numeric(stock_df['Close'], errors='coerce').ffill()
    
    # Calculate moving average
    moving_avg = calculate_moving_avg(close, length, average_type)

    # Define conditions
    long_entry = close > moving_avg * (1 + entry_percent / 100)
    long_exit = close < moving_avg * (1 - exit_percent / 100)
    short_entry = close < moving_avg * (1 - entry_percent / 100)
    short_exit = close > moving_avg * (1 + exit_percent / 100)

    # Create signal columns
    signals['trend_following_buy_signals'] = 'neutral'
    signals['trend_following_sell_signals'] = 'neutral'

    # Assign 'buy' signals
    signals.loc[long_entry | short_exit, 'trend_following_buy_signals'] = 'buy'

    # Assign 'sell' signals
    signals.loc[long_exit | short_entry, 'trend_following_sell_signals'] = 'sell'

    return signals



# Functions from universaloscillatorstrat.py
import numpy as np
import pandas as pd

def universal_oscillator_strategy(data, cutoff_length=20, mode='reversal'):
    signals = pd.DataFrame(index=data.index)
    close = np.array(data['Close'])
    osc = ehlers_universal_oscillator(close, cutoff_length)
    signals['osc'] = osc
    
    if mode == 'trend_following':
        signals['universal_oscillator_strategy_signals'] = 'neutral'
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_signals'] = 'long'
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_signals'] = 'short'
    elif mode == 'reversal':
        signals['universal_oscillator_strategy_signals'] = 'neutral'
        signals.loc[(signals['osc'] < 0) & (signals['osc'].shift(1) >= 0), 'universal_oscillator_strategy_signals'] = 'long'
        signals.loc[(signals['osc'] > 0) & (signals['osc'].shift(1) <= 0), 'universal_oscillator_strategy_signals'] = 'short'
    
    signals.drop(['osc'], axis=1, inplace=True)
    return signals

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


# Functions from vhftrend.py
import pandas as pd
import numpy as np
import talib

# VHF Trend Strategy
def vhf_signals(stock_df, length=14, lag=14, avg_length=14, trend_level=0.5, max_level=0.75, crit_level=0.25, mult=2, avg_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the VHF indicator
    vhf = (np.max(stock_df['High'] - stock_df['Low']) / (stock_df['Close'].diff().abs().rolling(length).sum())) * 100
    
    # Get Lowest VHF value in previous period
    min_vhf = vhf.rolling(lag).min()

    # Get Moving Average
    if avg_type.lower() == 'simple':
        ma = stock_df['Close'].rolling(avg_length).mean()
    elif avg_type.lower() == 'exponential':
        ma = stock_df['Close'].ewm(span=avg_length).mean()
    elif avg_type.lower() == 'weighted':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    elif avg_type.lower() == 'wilders':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    elif avg_type.lower() == 'hull':
        ma = talib.WMA(stock_df['Close'].values, timeperiod=avg_length)
    else:
        raise ValueError("Invalid average type")

    signals['vhf_signal'] = 'neutral'

    # Generate Buy and Sell signals
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] > ma), 'vhf_signal'] = 'buy-to-open'
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] > ma), 'vhf_signal'] = 'buy-to-open'
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] < ma), 'vhf_signal'] = 'sell-to-open'
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] < ma), 'vhf_signal'] = 'sell-to-open'

    # Generate Close signals
    signals.loc[(stock_df['Close'] < ma) & (signals['vhf_signal'] == 'buy-to-open'), 'vhf_signal'] = 'sell-to-close'
    signals.loc[(stock_df['Close'] > ma) & (signals['vhf_signal'] == 'sell-to-open'), 'vhf_signal'] = 'buy-to-close'
    
    return signals


# Functions from volatilityband.py
import pandas as pd
import numpy as np
from ta import volatility

def volatility_band_signals(df, price_col='Close', average_length=20, deviation=2, low_band_adjust=0.5):
    
    # Extract the actual price column
    price = df[price_col]

    # Calculate SMA of price data
    mid_line = price.rolling(average_length).mean()
    
    # Create Bollinger Bands Indicator
    indicator_bb = volatility.BollingerBands(close=price, window=average_length, window_dev=deviation)

    # Get upper and lower bands
    df['upper_band'] = indicator_bb.bollinger_hband()
    df['lower_band'] = mid_line - ((indicator_bb.bollinger_hband() - mid_line) * low_band_adjust)

    # Create signals DataFrame
    signals = pd.DataFrame(index=df.index)
    signals['volatility_band_signals'] = 'neutral'

    # Create buy and sell signals
    signals.loc[price > df['upper_band'], 'volatility_band_signals'] = 'sell'
    signals.loc[price < df['lower_band'], 'volatility_band_signals'] = 'buy'

    return signals


# Functions from volswitch.py
import pandas as pd
from talib import MA_Type
import talib

# VolSwitch Strategy
def vols_switch_signals(stock_df, length=14, rsi_length=14, avg_length=50, rsi_overbought_level=70, rsi_oversold_level=30, rsi_average_type=MA_Type.SMA):
    # Calculate volatility using standard deviation
    stock_df['volatility'] = stock_df['Close'].rolling(window=length).std()
    
    # Calculate SMA
    stock_df['SMA'] = talib.MA(stock_df['Close'], timeperiod=avg_length, matype=rsi_average_type)

    # Calculate RSI
    stock_df['RSI'] = talib.RSI(stock_df['Close'], timeperiod=rsi_length)

    # Buy signal based on VolSwitch strategy
    stock_df['Buy_Signal'] = ((stock_df['volatility'].shift(1) > stock_df['volatility']) & 
                              (stock_df['Close'] > stock_df['SMA']) & 
                              (stock_df['RSI'] > rsi_oversold_level))

    # Sell signal based on VolSwitch strategy
    stock_df['Sell_Signal'] = ((stock_df['volatility'].shift(1) < stock_df['volatility']) & 
                               (stock_df['Close'] < stock_df['SMA']) & 
                               (stock_df['RSI'] < rsi_overbought_level))
    
    signals = pd.DataFrame(index=stock_df.index)
    signals['vols_switch_signal'] = 'neutral'
    
    # Create signals
    signals.loc[(stock_df['Buy_Signal'] == True), 'vols_switch_signal'] = 'long'
    signals.loc[(stock_df['Sell_Signal'] == True), 'vols_switch_signal'] = 'short'
 
    return signals

# Functions from voltyexpancloselx.py
import pandas as pd
import talib
from talib import MA_Type

def volty_expan_close_lx(df, num_atrs=2, length=14, ma_type='SMA'):
    df = df.reset_index()  # ensure that the Dataframe index is sequential
    atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=length)

    # Get the MA_Type based on the average type input
    if ma_type == 'SMA':
        ma_type = MA_Type.SMA
    elif ma_type == 'EMA':
        ma_type = MA_Type.EMA
    elif ma_type == 'WMA':
        ma_type = MA_Type.WMA
    elif ma_type == 'DEMA':
        ma_type = MA_Type.DEMA
    elif ma_type == 'TEMA':
        ma_type = MA_Type.TEMA
    elif ma_type == 'TRIMA':
        ma_type = MA_Type.TRIMA
    elif ma_type == 'KAMA':
        ma_type = MA_Type.KAMA
    elif ma_type == 'MAMA':
        ma_type = MA_Type.MAMA
    else:
        ma_type = MA_Type.SMA

    atr_ma = talib.MA(atr, timeperiod=length, matype=ma_type)

    df['volty_expan_close_lx_signal'] = 0
    for i in range(length, len(df)):
        if df.loc[i, 'Low'] < (df.loc[i-1, 'Close'] - num_atrs * atr_ma[i]):
            df.loc[i, 'volty_expan_close_lx_signal'] = -1 if df.loc[i, 'Open'] < (df.loc[i-1, 'Close'] - num_atrs * atr_ma[i]) else 0

    return df[['volty_expan_close_lx_signal']]



# Functions from vpnstrat.py
import pandas as pd
import numpy as np
from ta import momentum, volatility, volume

def vpn_signals(df, length=14, ema_length=14, average_length=10, rsi_length=14, volume_average_length=50, highest_length=14, atr_length=14, factor=1, critical_value=10, rsi_max_overbought_level=90, num_atrs=1, average_type='simple'):
    vwap = volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=average_length)

    # Calculate VPN 
    vpn = (df['Volume'] * factor * np.where(df['Close'].diff() > 0, 1, -1)).ewm(span=length).mean() / df['Volume'].ewm(span=length).mean()
    vpn_average = vpn.ewm(span=average_length).mean() if average_type == 'exponential' else vpn.rolling(window=average_length).mean()
    
    # Create signal column
    signals = pd.DataFrame(index=df.index)
    signals['vpn_signals'] = 'neutral'

    # Buy condition
    buy_condition = (
        (vpn > critical_value) & 
        (df['Close'] > vwap.vwap) &  # Fixed incorrect function call
        (df['Volume'].rolling(window=volume_average_length).mean().diff() > 0) & 
        (momentum.RSIIndicator(close=df['Close'], window=rsi_length).rsi() < rsi_max_overbought_level)
    )

    # Sell condition
    sell_condition = (
        (vpn < vpn_average) & 
        (df['Close'] < df['Close'].rolling(window=highest_length).max() - num_atrs * 
         volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=atr_length).average_true_range())
    )

    # Assign buy and sell signals
    signals.loc[buy_condition, 'vpn_signals'] = 'buy'
    signals.loc[sell_condition, 'vpn_signals'] = 'sell'

    return signals


# Functions from vwmabreakouts.py
import pandas as pd
import numpy as np
from ta import volume

# VWMA Breakouts Strategy
def vwma_breakouts(stock_df, vwma_length=50, sma_length=70):
    
    df = pd.DataFrame(index=stock_df.index)
    
    # Create the VWAP indicator object
    vwap = volume.VolumeWeightedAveragePrice(
        high = stock_df['High'],
        low = stock_df['Low'],
        close = stock_df['Close'],
        volume = stock_df['Volume'],
        window = vwma_length
    )
    
    # Calculate the VWMA
    df['VWMA'] = vwap.volume_weighted_average_price()

    # Calculate the SMA
    df['SMA'] = df['VWMA'].rolling(window=sma_length).mean()

    # Define signals
    df['vwma_breakouts_signal'] = 'neutral'
    df.loc[df['VWMA'] > df['SMA'], 'vwma_breakouts_signal'] = 'long'
    df.loc[df['VWMA'] < df['SMA'], 'vwma_breakouts_signal'] = 'short'

    return df['vwma_breakouts_signal']
