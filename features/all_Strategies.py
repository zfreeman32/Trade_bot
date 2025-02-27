#%%
import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import talib
import datetime
from talib import MA_Type

#%%
# Functions from accumulationdistributionstrat.py
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
    signals['acc_dist_buy_signal'] = np.where(
        (df['range_ratio'] < factor) &
        df['higher_low'] &
        df['break_high'] &
        (df['vol_ratio_av'] > vol_ratio),
        1, 0)
    
    # Sell signals
    signals['acc_dist_sell_signal'] = np.where(
        df['fall_below'],
        1, 0)
    return signals

#%%
# Functions from adxbreakoutsle.py
def adx_breakouts_signals(stock_df, highest_length=15, adx_length=14, adx_level=40, offset=0.5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Compute the rolling highest high and align with stock_df
    highest = stock_df['High'].rolling(window=highest_length).max()
    
    # Compute ADX and align it with stock_df
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_length).adx()
    
    # Ensure indexes are aligned
    signals['adx'] = adx
    signals['highest'] = highest
    signals['adx_breakout_buy_signal'] = 0

    # Handle potential NaNs by forward-filling
    signals.fillna(method='bfill', inplace=True)
    
    # Compute the breakout condition
    breakout_condition = (signals['adx'] > adx_level) & (stock_df['Close'] > (signals['highest'] + offset))
    
    signals.loc[breakout_condition, 'adx_breakout_buy_signal'] = 1
    
    # Drop temporary columns
    signals.drop(['adx', 'highest'], axis=1, inplace=True)

    return signals

#%%
# Functions from adxtrend.py
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

#%%
# Functions from atrhighsmabreakoutsle.py
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

#%%
# Functions from atrtrailingstople.py
def atr_trailing_stop_le_signals(stock_df, atr_period=14, atr_factor=3):
    close = stock_df['Close']
    high = stock_df['High']
    low = stock_df['Low']
    
    atr = volatility.AverageTrueRange(high, low, close, window=atr_period).average_true_range()
    atr_trailing_stop = close - atr_factor * atr

    signals = pd.DataFrame(index=stock_df.index)
    signals['atr_trailing_stop'] = atr_trailing_stop
    signals['atr_trailing_stop_le_buy_signal'] = np.where(
        (close.shift(1) <= signals['atr_trailing_stop'].shift(1)) &  # Previous close was below ATR
        (close > signals['atr_trailing_stop']),  # Current close crosses above ATR
        1,  # Signal triggered
        0   # No signal
    )

    signals.drop(['atr_trailing_stop'], axis=1, inplace=True)

    return signals

#%%
# Functions from atrtrailingstopse.py
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
        (close.shift(1) >= signals['atr_trailing_stop']) &  # Previous close was above ATR
        (close < signals['atr_trailing_stop']),  # Current close crosses below ATR
        1,  # Signal triggered
        0   # No signal
    )

    # Drop the internal 'atr_trailing_stop' column, as it is not part of the output
    signals.drop(columns=['atr_trailing_stop'], inplace=True)

    return signals

#%%
# Functions from bbdivergencestrat.py
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

#%%
# Functions from bollingerbandsle.py
def bollinger_bands_le_signals(data, length=20, num_devs_dn=2.0):
    # Instantiate Bollinger Bands indicator
    indicator_bb = volatility.BollingerBands(close=data['Close'], window=length, window_dev=num_devs_dn)
    
    # Create DataFrame for signals
    signals = pd.DataFrame(index=data.index)
    signals['lower'] = indicator_bb.bollinger_lband()  # Calculate lower band
    signals['close'] = data['Close']  # Track closing prices
    signals['bollinger_bands_le_buy_signal'] = 0 # Default all signals to 0.0

    # Generate 'Long Entry' signal where price crosses above lower band
    signals.loc['bollinger_bands_le_buy_signal'][signals['close'] > signals['lower'].shift(1)] = 1
    signals.drop(columns=['close', 'lower'], inplace=True)
    # Return only the signal column
    return signals

#%%
# Functions from bollingerbandsse.py
def bollinger_bands_short_entry(df, length=20, num_devs_up=2):
    # Calculate Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=df["Close"], window=length, window_dev=num_devs_up)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()
    # Generate signals
    df['bb_short_entry_signal'] = np.where(df['Close'] < df['bb_bbh'], 1, 0)
    signals = pd.DataFrame(df['bb_short_entry_signal'])
    return signals[['bb_short_entry_signal']]

#%%
# Functions from camarillapointsstrat.py
def camarilla_strategy(stock_df):
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

    signals = pd.DataFrame(index=stock_df.index)
    signals['camarilla_buy_signal'] = 0
    signals['camarilla_sell_signal'] = 0
  
    # Long entry
    signals.loc[((stock_df.Open < stock_df.S3) & (stock_df.Close > stock_df.S3)), 'camarilla_buy_signal'] = 1
    # Short entry
    signals.loc[((stock_df.Open > stock_df.R3) & (stock_df.Close < stock_df.R3)), 'camarilla_sell_signal'] = 1

    return signals[['camarilla_buy_signal', 'camarilla_sell_signal']]

#%%
# Functions from consbarsdownse.py
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

#%%
# Functions from consbarsuple.py
def cons_bars_up_le_signals(stock_df, consec_bars_up=4, price='Close'):
    # Initialize the signal DataFrame with the same index as stock_df
    signals = pd.DataFrame(index=stock_df.index)
    signals['Signal'] = 0

    # Create a boolean mask indicating where price bars are consecutively increasing
    mask = stock_df[price].diff() > 0
    
    # Calculate the rolling sum of increasing bars
    rolling_sum = mask.rolling(window=consec_bars_up).sum()
    
    # Identify where the count of increasing bars exceeds the threshold
    signals.loc[rolling_sum >= consec_bars_up, 'Signal'] = 1

    # Take the difference of signals to capture the transition points
    signals['Signal'] = signals['Signal'].diff().fillna(0)

    # Initialize column to ensure integer type
    signals['cons_bars_up_le_buy_signal'] = 0  

    # Assign buy signal only at valid transition points
    signals.loc[signals['Signal'] > 0, 'cons_bars_up_le_buy_signal'] = 1

    # Drop intermediate column
    signals.drop(columns=['Signal'], inplace=True)

    # Ensure the output is integer type (avoids float 1.0 and NaN)
    signals['cons_bars_up_le_buy_signal'] = signals['cons_bars_up_le_buy_signal'].astype(int)

    return signals

#%%
# Donchain Channel Signals
def donchian_signals(df, entry_length=40, exit_length=15, atr_length=20, atr_factor=2, atr_stop_factor=2, atr_average_type='simple'):
    signals = pd.DataFrame(index=df.index)

    # Ensure numeric conversion
    df['High'] = pd.to_numeric(df['High'], errors='coerce').ffill()
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce').ffill()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce').ffill()

    # Compute Donchian Channel Stops (ensuring proper alignment)
    signals['BuyStop'] = df['High'].rolling(window=entry_length, min_periods=1).max()
    signals['ShortStop'] = df['Low'].rolling(window=entry_length, min_periods=1).min()
    signals['CoverStop'] = df['High'].rolling(window=exit_length, min_periods=1).max()
    signals['SellStop'] = df['Low'].rolling(window=exit_length, min_periods=1).min()

    # Compute ATR
    atr = volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_length).average_true_range()

    # Apply chosen smoothing type
    if atr_average_type == "simple":
        atr_smoothed = atr.rolling(window=atr_length, min_periods=1).mean()
    elif atr_average_type == "exponential":
        atr_smoothed = atr.ewm(span=atr_length, adjust=False).mean()
    elif atr_average_type == "wilder":
        atr_smoothed = atr.ewm(alpha=1/atr_length, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported ATR smoothing type '{atr_average_type}'")

    # Initialize signal columns
    signals['donchian_buy_signals'] = 0
    signals['donchian_sell_signal'] = 0

    # Apply ATR filter if necessary
    if atr_factor > 0:
        volatility_filter = atr_smoothed * atr_factor

        # Ensure all Series align properly
        buy_condition = (df['High'] > signals['BuyStop']) & ((df['High'].shift(1) - volatility_filter.shift(1)) > 0)
        sell_condition = (df['Low'] < signals['ShortStop']) & ((df['Low'].shift(1) - volatility_filter.shift(1)) < 0)

        signals.loc[buy_condition, 'donchian_buy_signals'] = 1
        signals.loc[sell_condition, 'donchian_sell_signal'] = 1
    else:
        signals.loc[df['High'] > signals['BuyStop'], 'donchian_buy_signals'] = 1
        signals.loc[df['Low'] < signals['ShortStop'], 'donchian_sell_signal'] = 1

    # Drop unnecessary columns
    signals.drop(columns=['CoverStop', 'SellStop', 'BuyStop', 'ShortStop'], inplace=True)

    return signals.fillna(0)

#%%
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
            hp[i] = ((1 - a / 2) ** 2) * (src[i] - 2 * src[i-1] + src[i-2]) + \
                2 * (1 - a) * hp[i-1] - ((1 - a) ** 2) * hp[i-2]
        return hp
    
    # Compute the Ehlers Decycler Oscillator
    hp1 = hp_filter(price, length1)
    hp2 = hp_filter(price, length2)
    dec = hp2 - hp1
    
    # Compute signal strength
    slo = dec - np.roll(dec, 1)
    sig = np.where(slo > 0, np.where(slo > np.roll(slo, 1), 2, 1),
                   np.where(slo < 0, np.where(slo < np.roll(slo, 1), -2, -1), 0))
    
    # Generate separate buy and sell signals
    signals['ehlers_stoch_buy_signal'] = np.where((sig == 1) | (sig == 2), 1, 0)
    signals['ehlers_stoch_sell_signal'] = np.where((sig == -1) | (sig == -2), 1, 0)
    
    return signals

#%%
# Functions from eightmonthavg.py
def eight_month_avg_signals(stock_df, length=8):
    signals = pd.DataFrame(index=stock_df.index)  # Ensure index matches stock_df
    signals['sma'] = stock_df['Close'].rolling(window=length).mean()
    signals['eight_month_avg_buy_signal'] = 0
    signals['eight_month_avg_sell_signal'] = 0

    buy_condition = (stock_df['Close'] > signals['sma']) & (stock_df['Close'].shift(1) <= signals['sma'].shift(1))
    sell_condition = (stock_df['Close'] < signals['sma']) & (stock_df['Close'].shift(1) >= signals['sma'].shift(1))

    # Ensure the conditions' index matches stock_df.index before applying
    buy_condition = buy_condition.reindex(stock_df.index, fill_value=False)
    sell_condition = sell_condition.reindex(stock_df.index, fill_value=False)

    signals.loc[buy_condition, 'eight_month_avg_buy_signal'] = 1
    signals.loc[sell_condition, 'eight_month_avg_sell_signal'] = 1

    signals.drop(['sma'], axis=1, inplace=True)
    return signals

#%%
# Functions from fourdaybreakoutle.py
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
    signals['four_day_breakout_le_buy_signal'] = 0
    
    # Create a buy signal when conditions are met
    signals.loc[(stock_df['Close'] > stock_df['SMA']) &
                (stock_df['Bullish_Count'] == pattern_length) &
                (stock_df['Close'] > (max_pattern_high + breakout_amount)), 'four_day_breakout_le_buy_signal'] = 1
    
    return signals

#%%
# Functions from gandalfprojectresearchsystem.py
def gandalf_signals(df, exit_length=10, gain_exit_length=20):  
    signals = pd.DataFrame(index=df.index)

    # Calculating ohlc4, median price, and mid-body price
    df['ohlc4'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['median_price'] = (df['High'] + df['Low']) / 2
    df['mid_body'] = (df['Open'] + df['Close']) / 2

    # Initialize buy/sell signals as integer 0
    signals['gandalf_buy_signal'] = 0  
    signals['gandalf_sell_signal'] = 0  

    # Buy signal logic
    buy_condition = (
        ((df['ohlc4'].shift(1) < df['median_price'].shift(1)) &
         (df['median_price'].shift(2) <= df['ohlc4'].shift(1)) &
         (df['median_price'].shift(2) <= df['ohlc4'].shift(3))) |
        ((df['ohlc4'].shift(1) < df['median_price'].shift(3)) &
         (df['mid_body'] < df['median_price'].shift(2)) &
         (df['mid_body'].shift(1) < df['mid_body'].shift(2)))
    )
    signals.loc[buy_condition, 'gandalf_buy_signal'] = 1

    # Identify buy timestamps
    buy_indices = signals.index[signals['gandalf_buy_signal'] == 1]

    for buy_index in buy_indices:
        buy_timestamp = signals.index[signals.index == buy_index]
        
        if not buy_timestamp.empty:
            buy_timestamp = buy_timestamp[0]  # Get the timestamp
            
            if buy_timestamp in df.index:
                buy_open_price = df.at[buy_timestamp, 'Open']  # Scalar value access
                
                sell_condition = (
                    (signals.index >= buy_timestamp + pd.Timedelta(days=exit_length)) |
                    ((signals.index >= buy_timestamp + pd.Timedelta(days=gain_exit_length)) & 
                     (df['Close'] > buy_open_price)) |
                    ((df['Close'] < buy_open_price) &
                     (((df['ohlc4'].shift(-1) < df['mid_body'].shift(-1)) &
                       (df['median_price'].shift(-2) == df['mid_body'].shift(-3)) &
                       (df['mid_body'].shift(-1) <= df['mid_body'].shift(-4))) |
                      ((df['ohlc4'].shift(-2) < df['mid_body']) &
                       (df['median_price'].shift(-4) < df['ohlc4'].shift(-3)) &
                       (df['mid_body'].shift(-1) < df['ohlc4'].shift(-1)))))
                )
                
                signals.loc[sell_condition, 'gandalf_sell_signal'] = 1

    # Ensure integer output to avoid float conversion
    signals['gandalf_buy_signal'] = signals['gandalf_buy_signal'].astype(int)
    signals['gandalf_sell_signal'] = signals['gandalf_sell_signal'].astype(int)

    return signals

#%%
# GapDownSE Strategy
def gap_down_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['high'] = stock_df["High"]
    signals['prev_low'] = stock_df["Low"].shift(1)
    signals['gap_down_se_sell_signals'] = np.where(signals['high'] < signals['prev_low'], 1, 0)
    signals.drop(['high', 'prev_low'], axis=1, inplace=True)
    return signals

#%%
# Functions from gapmomentumsystem.py
def gap_momentum_signals(data_df, length=14, signal_length=9, full_range=False):
    signals = pd.DataFrame(index=data_df.index)
    # calculate the gap prices
    gaps = data_df['Open'] - data_df['Close'].shift()
    # calculate the signal line 
    signals['gap_avg'] = gaps.rolling(window=length).mean() 
    # Calculate the signal based on the gap average
    signals['gap_momentum_buy_signal'] = 0
    signals['gap_momentum_sell_signal'] = 0
    signals.loc[(signals['gap_avg'] > signals['gap_avg'].shift()), 'gap_momentum_buy_signal'] = 1
    signals.loc[(signals['gap_avg'] < signals['gap_avg'].shift()), 'gap_momentum_sell_signal'] = 1
    signals.drop(['gap_avg'], axis=1, inplace=True)
    return signals

#%%
# Functions from gapreversalle.py
def gap_reversal_signals(stock_df, gap=0.10, offset=0.50):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the gap from the previous day's low
    stock_df['PrevLow'] = stock_df['Low'].shift(1)
    stock_df['Gap'] = (stock_df['Open'] - stock_df['PrevLow']) / stock_df['PrevLow']

    # Calculate the offset from the gap
    stock_df['Offset'] = (stock_df['High'] - stock_df['Open'])
    
    # Generate signals based on condition
    signals['gap_reversal_buy_signal'] = 0
    signals.loc[(stock_df['Gap'] > gap) & (stock_df['Offset'] > offset), 'gap_reversal_buy_signal'] = 1
    
    return signals

#%%
# Functions from gapuple.py
def gap_up_le_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['gap_up_le_buy_signal'] = 0
    
    # Identify rows where the current Low is higher than the previous High
    gap_up = (stock_df['Low'] > stock_df['High'].shift(1))
    
    # Shift the signal to the next row
    shifted_gap_up = gap_up.shift(-1).fillna(False).astype(bool)  # Fill NaN with False to avoid indexer errors
    
    # Generate Long Entry signal for the next bar
    signals.loc[shifted_gap_up, 'gap_up_le_buy_signal'] = 1
    
    return signals

#%%
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
    signals['golden_cross_buy_signal'] = 0
    signals['golden_cross_sell_signal'] = 0
    signals.loc[signals['FastMA'] > signals['SlowMA'], 'golden_cross_buy_signal'] = 1
    signals.loc[signals['FastMA'] < signals['SlowMA'], 'golden_cross_sell_signal'] = 1
    signals.drop(columns=['FastMA', 'SlowMA'], inplace=True)

    return signals

#%%
# Functions from goldentrianglele.py
def golden_triangle_signals(stock_df, average_length=50, confirm_length=20, volume_length=5):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate moving averages
    sma_long = trend.SMAIndicator(stock_df['Close'], average_length)
    sma_short = trend.SMAIndicator(stock_df['Close'], confirm_length)
    
    # Identify initial uptrend condition
    uptrend = stock_df['Close'] > sma_long.sma_indicator()
    
    # Identify pivot points as local maxima
    pivot = ((stock_df['Close'] > stock_df['Close'].shift()) &
             (stock_df['Close'] > stock_df['Close'].shift(-1)))
    
    # Identify price drop condition
    price_drop = stock_df['Close'] < sma_long.sma_indicator()
    
    # Define initial triangle setup condition
    triangle_setup = (uptrend & pivot & price_drop).shift().fillna(False)
    
    # Price and volume confirmation
    price_confirm = stock_df['Close'] > sma_short.sma_indicator()
    volume_confirm = stock_df['Volume'] > stock_df['Volume'].rolling(volume_length).max()
    triangle_confirm = (price_confirm & volume_confirm).shift().fillna(False)
    
    # Generate buy signals
    signals['golden_triangle_buy_signal'] = np.where(triangle_setup & triangle_confirm, 1, 0)
    
    return signals

#%%
# Functions from hacoltstrat.py
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
    
    # Create buy and sell signal columns
    signals['hacolt_buy_signal'] = np.where(hacolt == 100, 1, 0)
    signals['hacolt_sell_signal'] = np.where(hacolt == 0, 1, 0)
    
    return signals

#%%
# Functions from halloween.py
def halloween_strategy(data: pd.DataFrame, sma_length: int = 30):
    signals = pd.DataFrame(index=data.index)
    
    
    # Create SMA Indicator
    sma = trend.SMAIndicator(data["Close"], sma_length)
    signals['SMA'] = sma.sma_indicator()
    
    # Create a signal column and initialize it to Hold.
    signals['halloween_strategy_buy_signal'] = 0
    signals['halloween_strategy_sell_signal'] = 0
    
    # Generate Long Entry signal
    signals.loc[(signals.index.month == 10) & (signals.index.day == 1) & (data['Close'] > signals['SMA']), 'halloween_strategy_buy_signal'] = 1
    
    # Generate Long Exit Signal
    signals.loc[(signals.index.month == 5) & (signals.index.day == 1), 'halloween_strategy_sell_signal'] = 1
    signals.drop(['SMA'], axis=1, inplace=True)
    return signals

#%%
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
    
    # Create buy and sell signal columns
    signals['ift_stoch_buy_signal'] = np.where((signals['IFTStoch'] > over_sold) & (signals['IFTStoch'].shift(1) <= over_sold), 1, 0)
    signals['ift_stoch_sell_signal'] = np.where((signals['IFTStoch'] < over_bought) & (signals['IFTStoch'].shift(1) >= over_bought) & (df['Close'] < signals['SMA']), 1, 0)
    
    signals.drop(['IFTStoch', 'SMA'], axis=1, inplace=True)
    
    return signals

#%%
# Functions from insidebarle.py
def inside_bar_le(df):
    # Create a copy of the DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create a new column for the Long Entry signals
    signals['inside_bar_le_buy_signal'] = 0

    # Calculate the Inside Bar condition
    signals['Inside_Bar'] = (df['High'] < df['High'].shift(1)) & (df['Low'] > df['Low'].shift(1))
    
    # Generate the Long Entry signal condition
    signals.loc[signals['Inside_Bar'] & (df['Close'] > df['Open']), 'inside_bar_le_buy_signal'] = 1
    signals.drop(['Inside_Bar'], axis=1, inplace=True)
    return signals['inside_bar_le_buy_signal']

#%%
# Functions from insidebarse.py
# InsideBarSE Strategy
def inside_bar_se_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)

    # Identify the inside bars and close price is lower than open
    signals['inside_bar'] = np.where((stock_df['High'] < stock_df['High'].shift(1)) &
                                     (stock_df['Low'] > stock_df['Low'].shift(1)) &
                                     (stock_df['Close'] < stock_df['Open']), 1, 0)

    # Generate signals
    signals['inside_bar_sell_signal'] = 0
    signals.loc[(signals['inside_bar'].shift(1) == 1), 'inside_bar_sell_signal'] = 1
    signals.drop(['inside_bar'], axis=1, inplace=True)
    return signals

#%%
# Functions from keyrevle.py
def key_rev_le_signals(stock_df, length=5):
    """
    Generate KeyRevLE strategy signals.
    :param stock_df: OHLCV dataset.
    :param length: The number of preceding bars whose Low prices are compared to the current Low.
    :return: DataFrame with 'key_rev_le_signal' column.
    """
    signals = pd.DataFrame(index=stock_df.index)
    signals['key_rev_buy_signal'] = 0

    for i in range(length, len(stock_df) - 1):  # Avoid index error at the end
        if stock_df['Low'].iloc[i] < stock_df['Low'].iloc[i-length:i].min() and \
           stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i-1]:
            signals.loc[signals.index[i+1], 'key_rev_buy_signal'] = 1  # Fix using `.loc[]`
            
    return signals

#%%
# Functions from keyrevlx.py
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
    signals['key_rev_sell_signals'] = 0

    # Check the condition for each row
    for i in range(length, len(stock_df)):
        if stock_df['High'].iloc[i] > stock_df['High'].iloc[i-length:i].max() and \
           stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i-1]:
            signals.iloc[i, signals.columns.get_loc('key_rev_sell_signals')] = 1  # Use .iloc[] for safe modification

    return signals

#%%
# Functions from macdstrat.py
def macd_signals(df, fast_length=12, slow_length=26, macd_length=9):
    # Create a signals DataFrame
    signals = pd.DataFrame(index=df.index)

    # Create MACD indicator
    macd = trend.MACD(df['Close'], window_slow=slow_length, window_fast=fast_length, window_sign=macd_length)

    # Generate MACD line and signal line
    signals['MACD_line'] = macd.macd()
    signals['MACD_signal'] = macd.macd_signal()
    
    # Create a column for the macd strategy signal
    signals['macd_strat_buy_signal'] = 0
    signals['macd_strat_sell_signal'] = 0

    # Create signals: When the MACD line crosses the signal line upward, buy the stock
    signals['macd_strat_buy_signal'][(signals['MACD_line'] > signals['MACD_signal']) & (signals['MACD_line'].shift(1) < signals['MACD_signal'].shift(1))] = 1
    
    # When the MACD line crosses the signal line downward, sell the stock
    signals['macd_strat_sell_signal'][(signals['MACD_line'] < signals['MACD_signal']) & (signals['MACD_line'].shift(1) > signals['MACD_signal'].shift(1))] = 1
    signals.drop(['MACD_line','MACD_signal'], axis=1, inplace=True)
    return signals

#%%
# Functions from momentumle.py
def momentumle_signals(stock_df, length=12, price_scale=100):
    # Initialize the signals DataFrame
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate Momentum
    mom = momentum.roc(stock_df['Close'], window=length)
    
    # Create buy and sell signal columns
    signals['momentumle_buy_signal'] = np.where((mom > 0) & (mom.shift(1) <= 0), 1, 0)
    
    # Calculate signal price level
    signal_price_level = stock_df['High'] + (1 / price_scale)
    
    # Generate buy signals if price level condition is met
    signals['momentumle_buy_signal'] = np.where(stock_df['Open'].shift(-1) > signal_price_level, 1, signals['momentumle_buy_signal'])
        
    return signals

#%%
# Functions from movavgstrat.py
def moving_average_strategy(df, window=15, average_type='simple', mode='trend Following'):
    # Compute moving average
    if average_type == 'simple':
        df['moving_avg'] = df['Close'].rolling(window=window).mean()
    elif average_type == 'exponential':
        df['moving_avg'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Create buy and sell signal columns
    df['moving_average_buy_signal'] = np.where(df['Close'] > df['moving_avg'], 1, 0) if mode == 'trend Following' else np.where(df['Close'] < df['moving_avg'], 1, 0)
    df['moving_average_sell_signal'] = np.where(df['Close'] < df['moving_avg'], 1, 0) if mode == 'trend Following' else np.where(df['Close'] > df['moving_avg'], 1, 0)
    
    return df[['moving_average_buy_signal', 'moving_average_sell_signal']]

#%%
# Functions from movavgtwolinesstrat.py
def mov_avg_two_lines_signals(stock_df, fast_length=5, slow_length=20, average_type='EMA', strategy_name='mov_avg_two_lines'):
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
        fastMA = talib.WMA(price, timeperiod=fast_length)  # Hull is not directly supported by TA-Lib
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
        slowMA = talib.WMA(price, timeperiod=slow_length)  # Hull is not directly supported by TA-Lib
    else:
        slowMA = talib.EMA(price, timeperiod=slow_length)

    buy_signal_col = 'mov_avg_two_lines_buy_signal'
    sell_signal_col = 'mov_avg_two_lines_sell_signal'
    
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    signals.loc[(fastMA > slowMA) & (fastMA.shift(1) <= slowMA.shift(1)), buy_signal_col] = 1
    signals.loc[(fastMA < slowMA) & (fastMA.shift(1) >= slowMA.shift(1)), sell_signal_col] = 1

    return signals

#%%
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
    signals['pmo_buy_signal'] = np.where((signals['pmo_line'] > signals['pmo_signals']) & (signals['pmo_line'].shift(1) <= signals['pmo_signals'].shift(1)), 1, 0)
    signals['pmo_sell_signal'] = np.where((signals['pmo_line'] < signals['pmo_signals']) & (signals['pmo_line'].shift(1) >= signals['pmo_signals'].shift(1)), 1, 0)
    
    signals.drop(['pmo_line', 'pmo_signals'], axis=1, inplace=True)

    return signals

#%%
# Functions from priceswing.py
def price_swing_signals(stock_df, swing_type="RSI", length=20, exit_length=20, deviations=2, overbought=70, oversold=30):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the Bollinger Bands
    indicator_bb = volatility.BollingerBands(close=stock_df['Close'], window=length, window_dev=deviations)
    stock_df['bb_bbm'] = indicator_bb.bollinger_mavg()
    stock_df['bb_bbh'] = indicator_bb.bollinger_hband()
    stock_df['bb_bbl'] = indicator_bb.bollinger_lband()

    buy_signal_col = 'price_swing_buy_signal'
    sell_signal_col = 'price_swing_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    if swing_type == "bollinger":
        # Use Bollinger Bands crossover swing type
        signals.loc[stock_df['Close'] > stock_df['bb_bbh'], sell_signal_col] = 1
        signals.loc[stock_df['Close'] < stock_df['bb_bbl'], buy_signal_col] = 1

    elif swing_type == "RSI":
        # Use RSI crossover swing type
        rsi = momentum.RSIIndicator(close=stock_df['Close'], window=length)
        signals['rsi'] = rsi.rsi()
        signals.loc[signals['rsi'] > overbought, sell_signal_col] = 1
        signals.loc[signals['rsi'] < oversold, buy_signal_col] = 1
        signals.drop(['rsi'], axis=1, inplace=True)
        
    return signals

#%%
# Price Zone Oscillator (PZO) Strategy
def pzo_signals(stock_df, length=14, ema_length=60):
    signals = pd.DataFrame(index=stock_df.index)
    pzo = ((stock_df['Close'] - stock_df['Close'].rolling(window=length).mean()) / stock_df['Close'].rolling(window=length).std())*100
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=length).adx()
    ema = stock_df['Close'].ewm(span=ema_length).mean()

    buy_signal_col = 'pzo_buy_signal'
    sell_signal_col = 'pzo_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    # ADX > 18, price > EMA, and PZO cross "-40" level or surpass "+15" level from below
    signals.loc[(adx > 18) & (stock_df['Close'] > ema) & (
        (pzo.shift(1) < -40) & (pzo > -40) |
        ((pzo.shift(1) < 0) & (pzo > 0) & (pzo > 15))), buy_signal_col] = 1
    
    # ADX < 18, and PZO cross "-40" or "+15" level from below
    signals.loc[(adx <= 18) & (
        (pzo.shift(1) < -40) & (pzo > -40) |
        (pzo.shift(1) < 15) & (pzo > 15)), buy_signal_col] = 1

    return signals

#%%
# PriceZoneOscillatorLX Strategy
def pzo_lx_signals(df, length=14, ema_length=60, strategy_name='pzo_lx'):
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
    buy_signal_col = 'pzo_lx_buy_signal'
    sell_signal_col = 'pzo_lx_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0
    
    # Set conditions for Long Exit signals
    signals.loc[(adx > 18) & (pzo > 60) & (pzo < pzo.shift(1)), sell_signal_col] = 1  # PZO above +60 and going down in trending
    signals.loc[(adx > 18) & (df['Close'] < ema) & (pzo < 0), sell_signal_col] = 1    # PZO negative and price below EMA in trending
    signals.loc[(adx < 18) & (pzo.shift(1) > 40) & (pzo < 0) & (df['Close'] < ema), sell_signal_col] = 1  # PZO below zero with prior crossing +40 and price below EMA in non-trending
    signals.loc[(adx < 18) & (pzo.shift(1) < 15) & (pzo > -5) & (pzo < 40), sell_signal_col] = 1    # PZO failed to rise above -40, instead fell below -5 in non-trending
    
    return signals

#%%
# Functions from pricezoneoscillatorse.py
def pzo_trend_signals(stock_df, length=14, ema_length=60):
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
    signals['pzo_sell_signal'] = 0
    signals.loc[adx_trending & price_below_ema & (cross_above_40 | (cross_above_zero & fall_below_minus5)), 'pzo_sell_signal'] = 1
    signals.loc[adx_not_trending & (cross_above_40 | fall_below_minus5), 'pzo_sell_signal'] = 1

    # Drop unnecessary columns to return only signals
    signals.drop(['PZO', 'EMA', 'ADX'], axis=1, inplace=True)

    return signals

#%%
# Functions from pricezoneoscillatorsx.py
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
    signals['pzosx_buy_signal'] = 0

    # Calculate short_exit conditions based on PZO, ADX and EMA values
    for i in range(2, signals.shape[0]):
        if (signals['adx'].iloc[i] > 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                (signals['pzo'].iloc[i] > 0 and df['Close'].iloc[i] > signals['ema'].iloc[i])):
                signals.loc[signals.index[i], 'pzosx_buy_signal'] = 1
        elif (signals['adx'].iloc[i] < 18):
            if ((signals['pzo'].iloc[i] > -60 and signals['pzo'].iloc[i-1] <= -60) or
                ((signals['pzo'].iloc[i] > 0 or signals['pzo'].iloc[i-1] > -40) and df['Close'].iloc[i] > signals['ema'].iloc[i]) or
                (signals['pzo'].iloc[i] > 15 and signals['pzo'].iloc[i-1] <= -5 and signals['pzo'].iloc[i-2] > -40)):
                signals.loc[signals.index[i], 'pzosx_buy_signal'] = 1
        
    signals.drop(['pzo', 'ema', 'adx'], axis=1, inplace=True)

    return signals

#%%
# Functions from profittargetlx.py
def profit_target_lx_signals(stock_df, target=0.01, offset_type="percent"):
    signals = pd.DataFrame(index=stock_df.index)
    signals['profit_target_sell_signal'] = 0

    if offset_type=="percent":
        exit_price = stock_df['Close'] * (1 + target)
    elif offset_type=="tick":
        exit_price = stock_df['Close'] + (stock_df['Close'].diff() * target)
    elif offset_type=="value":
        exit_price = stock_df['Close'] + target
    else:
        return "Invalid offset type. Please use 'percent', 'tick' or 'value'."

    signals.loc[stock_df['Close'].shift(-1) >= exit_price, 'profit_target_sell_signal'] = 1

    return signals

#%%
# Functions from profittargetsx.py
def profit_target_SX(df, target=0.75, offset_type='value', tick_size=0.01):
    signals = pd.DataFrame(index=df.index)
    signals['profit_target_buy_signal'] = 0
    if offset_type == 'value':
        signals['profit_target_buy_signal'] = np.where(df['Close'].diff() <= -target, 'Short Exit', signals['profit_target_buy_signal'])
    elif offset_type == 'tick':
        signals['profit_target_buy_signal'] = np.where(df['Close'].diff() <= -(target * tick_size), 'Short Exit', signals['profit_target_buy_signal'])
    elif offset_type == 'percent':
        signals['profit_target_buy_signal'] = np.where(df['Close'].pct_change() <= -target/100, 'Short Exit', signals['profit_target_buy_signal'])  
    return signals

#%%
# Functions from rateofchangewithbandsstrat.py
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
    }
    signals['AvgROC'] = mov_avgs[average_type](signals['ROC'], window=average_length).mean()
    
    # Compute RMS of ROC
    signals['RMS'] = np.sqrt(np.mean(np.square(signals['ROC'].diff().dropna())))
    
    # Compute bands
    signals['LowerBand'] = signals['AvgROC'] - num_rmss * signals['RMS']
    signals['UpperBand'] = signals['AvgROC'] + num_rmss * signals['RMS']
    
    # Compute EMA
    signals['EMA'] = stock_df['Close'].ewm(span=ema_length, adjust=False).mean()
    
    # Initialize buy/sell signal columns
    buy_signal_col = 'rocwb_buy_signal'
    sell_signal_col = 'rocwb_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0
    
    # Generate Buy signals (when Close is above EMA and ROC is above LowerBand)
    signals.loc[(stock_df['Close'] > signals['EMA']) & (signals['ROC'] > signals['LowerBand']), buy_signal_col] = 1
    
    # Generate Sell signals (when Close is below EMA and ROC is below UpperBand)
    signals.loc[(stock_df['Close'] < signals['EMA']) & (signals['ROC'] < signals['UpperBand']), sell_signal_col] = 1
    
    # Remove all auxiliary columns
    signals.drop(['ROC', 'AvgROC', 'RMS', 'LowerBand', 'UpperBand', 'EMA'], axis=1, inplace=True)
    
    # Return signals DataFrame
    return signals

#%%
# Functions from rsistrat.py
def rsi_signals(df, length=14, overbought=70, oversold=30, rsi_average_type='simple'):
    close_price = df['Close']

    if rsi_average_type == 'simple':
        rsi = momentum.RSIIndicator(close_price, window=length).rsi()
    elif rsi_average_type == 'exponential':
        rsi = close_price.ewm(span=length, min_periods=length - 1).mean()

    signals = pd.DataFrame(index=df.index)
    signals['RSI'] = rsi

    signals['RSI_strat_buy_signal'] = 0
    signals['RSI_strat_sell_signal'] = 0
    signals.loc[(signals['RSI'] > oversold) & (signals['RSI'].shift(1) <= oversold), 'RSI_strat_buy_signal'] = 1
    signals.loc[(signals['RSI'] < overbought) & (signals['RSI'].shift(1) >= overbought), 'RSI_strat_sell_signal'] = 1
    signals.drop(['RSI'], axis=1, inplace=True)
    return signals

#%%
# Functions from simplerocstrat.py
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
    signals['roc_buy_signal'] = 0
    signals['roc_sell_signal'] = 0
    signals.loc[(signals['ROC'] > 0) & (signals['ROC'].shift(1) <= 0), 'roc_buy_signal'] = 0
    signals.loc[(signals['ROC'] < 0) & (signals['ROC'].shift(1) >= 0), 'roc_sell_signal'] = 0

    signals.drop(['ROC'], axis=1, inplace=True)
    return signals

#%%
# Functions from spectrumbarsle.py
# SpectrumBarsLE Strategy
def spectrum_bars_le_signals(df, length=10):
    signals = pd.DataFrame(index=df.index)
    signals['close_shift'] = df['Close'].shift(length)
    
    # Define the SpectrumBarsLE conditions:
    # Close price is greater than that from a specified number of bars ago 
    signals['spectrum_bars_buy_signal'] = np.where(df['Close'] > signals['close_shift'], 1, 0)

    # Drop unnecessary columns
    signals.drop(columns='close_shift', inplace=True)
    
    return signals

#%%
# Functions from stiffnessstrat.py
def StiffnessStrat(df, length=84, average_length=20, exit_length=84, num_dev=2, entry_stiffness_level=90, exit_stiffness_level=50, market_index='Close'):
    
    # Get stiffness and market trend
    sma = trend.SMAIndicator(df['Close'], window=int(average_length)).sma_indicator()
    bollinger = volatility.BollingerBands(df['Close'], window=int(average_length), window_dev=num_dev)
    upper_band = bollinger.bollinger_hband()
    condition = (df['Close'] > sma + upper_band)
    df['stiffness'] = condition.rolling(window=100).sum() / 100 * 100

    if market_index not in df.columns:
        print(f"Warning: {market_index} not found in dataset. Using 'Close' instead.")
        market_index = 'Close'

    df['ema'] = trend.EMAIndicator(df[market_index]).ema_indicator()
    uptrend = (df['ema'] > df['ema'].shift()) & (df['ema'].shift() > df['ema'].shift(2))
    df['uptrend'] = uptrend
    
    # Entry and exit conditions
    entry = (df['uptrend'] & (df['stiffness'] > entry_stiffness_level)).shift()
    exit = ((df['stiffness'] < exit_stiffness_level) | ((df['stiffness'].shift().rolling(window=exit_length).count() == exit_length))).shift()
    
    df['Buy_Signal'] = np.where(entry, 'buy', 'neutral') 
    df['Sell_Signal'] = np.where(exit, 'sell', 'neutral') 

    # Combine into one signal column
    df['stiffness_strat_buy_signal'] = 0
    df['stiffness_strat_sell_signal'] = 0
    df.loc[df['Buy_Signal'] == 'buy', 'stiffness_strat_buy_signal'] = 1
    df.loc[df['Sell_Signal'] == 'sell', 'stiffness_strat_sell_signal'] = 1

    return df[['stiffness_strat_sell_signal', 'stiffness_strat_buy_signal']]

#%%
# Functions from stochastic.py
def stochastic_signals(stock_df, k=14, d=3, overbought=80, oversold=20):
    signals = pd.DataFrame(index=stock_df.index)
    stoch = momentum.StochasticOscillator(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=k, smooth_window=d)
    
    signals['stoch_k'] = stoch.stoch()
    signals['stoch_d'] = stoch.stoch_signal()
    
    signals['stochastic_strat_buy_signal'] = 0
    signals['stochastic_strat_sell_signal'] = 0
    # Create signal when 'stoch_k' crosses above 'stoch_d'
    signals.loc[signals['stoch_k'] > signals['stoch_d'], 'stochastic_strat_buy_signal'] = 1
    # Create signal when 'stoch_k' crosses below 'stoch_d'
    signals.loc[signals['stoch_k'] < signals['stoch_d'], 'stochastic_strat_sell_signal'] = 1
    
    # Create states of 'overbought' and 'oversold'
    signals['overbought'] = signals['stoch_k'] > overbought
    signals['oversold'] = signals['stoch_k'] < oversold
    signals.drop(['stoch_k', 'stoch_d', 'overbought','oversold'], axis=1, inplace=True)

    return signals

#%%
# Functions from stoplosslx.py
def stop_loss_lx_signals(stock_df, offset_type="percent", stop=0.75):

    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_sell_signal'] = 0
            
    if offset_type.lower() == "value":
        stop_loss_price = stock_df['Close'].shift(1) - stop
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    if offset_type.lower() == "tick":
        stop_loss_price = stock_df['Close'].shift(1) - (stop * stock_df['TickSize'])
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    if offset_type.lower() == "percent":
        stop_loss_price = stock_df['Close'].shift(1) - (stock_df['Close'].shift(1) * stop/100)
        signals.loc[(stock_df['Close'] < stop_loss_price), 'stop_loss_sell_signal'] = 1
        
    return signals

#%%
# Functions from stoplosssx.py
def stop_loss_sx_signals(stock_df, offset_type='percent', stop=0.75):
    signals = pd.DataFrame(index=stock_df.index)
    signals['stop_loss_buy_signal'] = 0

    if offset_type.lower() == "value":
        signals.loc[(stock_df['Close'] - stock_df['Close'].shift() > stop), 'stop_loss_buy_signal'] = 1
    elif offset_type.lower() == "tick":
        tick_sizes = (stock_df['High'] - stock_df['Low']) / 2  # Assume the tick size is half the daily range
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / tick_sizes) > stop, 'stop_loss_buy_signal'] = 1
    elif offset_type.lower() == "percent":
        signals.loc[((stock_df['Close'] - stock_df['Close'].shift()) / stock_df['Close'].shift()) * 100 > stop, 'stop_loss_buy_signal'] = 1

    return signals

#%%
# Functions from svehatypcross.py
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
    signals['sve_ha_typ_cross_buy_signal'] = 0
    signals['sve_ha_typ_cross_sell_signal'] = 0

    # Assign 'buy' and 'sell' signals
    signals.loc[(tp_ema > ha_ema) & (tp_ema.shift() < ha_ema.shift()), 'sve_ha_typ_cross_buy_signal'] = 1
    signals.loc[(tp_ema < ha_ema) & (tp_ema.shift() > ha_ema.shift()), 'sve_ha_typ_cross_sell_signal'] = 1

    return signals[['sve_ha_typ_cross_buy_signal', 'sve_ha_typ_cross_sell_signal']]

#%%
# Functions from svesc.py
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
    signals['svesc_buy_signal'] = 0
    signals['svesc_sell_signal'] = 0
    signals.loc[signals['orders'] > 0 , 'svesc_buy_signal'] = 1
    signals.loc[signals['orders'] < 0 , 'svesc_sell_signal'] = 1
    
    # Drop unnecessary columns
    signals.drop(columns=['hlc3', 'ha_ohlc4', 'average_hlc3', 'average_ha_ohlc4', 'average_close', 'long_entry', 'short_entry', 'long_exit', 'short_exit', 'orders'], inplace=True)
    
    return signals

#%%
# Functions from svezlrbpercbstrat.py
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
    signals['sve_zl_rb_perc_buy_signal'] = 0
    signals['sve_zl_rb_perc_sell_signal'] = 0
    signals.loc[(signals['percent_b'] > signals['percent_b'].shift(1)) & (signals['stochastic'] > signals['stochastic'].shift(1)), 'sve_zl_rb_perc_buy_signal'] = 1
    signals.loc[(signals['percent_b'] < signals['percent_b'].shift(1)) & (signals['stochastic'] < signals['stochastic'].shift(1)), 'sve_zl_rb_perc_sell_signal'] = 1
    
    signals.drop(['percent_b', 'stochastic'], axis=1, inplace=True)

    return signals

#%%
# SwingThree Strategy
def swingthree_signals(stock_df, sma_length=14, ema_length=50, tick_sizes=5):
    signals = pd.DataFrame(index=stock_df.index)
    sma_high = trend.sma_indicator(stock_df['High'], sma_length)
    sma_low = trend.sma_indicator(stock_df['Low'], sma_length)
    ema_close = trend.ema_indicator(stock_df['Close'], ema_length)
    
    buy_signal_col = 'swingthree_buy_signal'
    sell_signal_col = 'swingthree_sell_signal'
    signals[buy_signal_col] = 0
    signals[sell_signal_col] = 0

    signals.loc[(stock_df['High'] > sma_high + tick_sizes) & (stock_df['Close'].shift(1) > ema_close), buy_signal_col] = 1
    signals.loc[(stock_df['Low'] < sma_low - tick_sizes) & (stock_df['Close'].shift(1) < ema_close), sell_signal_col] = 1
    
    return signals

#%%
# Functions from threebarinsidebarle.py
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
    signals['three_bar_inside_bar_buy_signal'] = np.where(conditions.shift(-1), 0, 0)

    return signals

#%%
# Functions from threebarinsidebarse.py
def three_bar_inside_bar_se(df):
    # Create a signal column initialized to 'neutral'
    signals = pd.DataFrame(index=df.index)
    signals['three_bar_inside_bar_sell_signal'] = 0
    
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
            signals.loc[df.index[i], 'three_bar_inside_bar_sell_signal'] = 1
    
    return signals

#%%
# Functions from trailingstoplx.py
def trailing_stop_lx_signals(stock_df, trail_stop=1, offset_type='percent'):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Stop Price'] = np.nan
    signals['trailing_stop_lx_sell_signal'] = 0

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
    signals.loc[stock_df['Low'] < signals['Stop Price'].shift(), 'trailing_stop_lx_sell_signals'] = 1

    return signals[['trailing_stop_lx_sell_signal']]

#%%
# Functions from trailingstopsx.py
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
    signals['trailing_stop_sx_buy_signal'] = 0
    signals.loc[(stock_df['High'] > signals['TrailingStop']), 'trailing_stop_sx_buy_signal'] = 1
    signals.drop(['TrailingStop'], axis=1, inplace=True)

    return signals[['trailing_stop_sx_buy_signal']]

#%%
# Functions from vhftrend.py
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

    signals['vhf_buy_signal'] = 0
    signals['vhf_sell_signal'] = 0

    # Generate Buy and Sell signals
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] > ma), 'vhf_buy_signal'] = 1
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] > ma), 'vhf_buy_signal'] = 1
    signals.loc[(vhf > crit_level) & (vhf > min_vhf*mult) & (stock_df['Close'] < ma), 'vhf_sell_signal'] = 1
    signals.loc[(vhf > trend_level) & (vhf < max_level) & (stock_df['Close'] < ma), 'vhf_sell_signal'] = 1

    return signals

#%%
# Functions from volatilityband.py
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
    signals['volatility_band_buy_signal'] = 0
    signals['volatility_band_sell_signal'] = 0

    # Create buy and sell signals
    signals.loc[price > df['upper_band'], 'volatility_band_sell_signal'] = 1
    signals.loc[price < df['lower_band'], 'volatility_band_buy_signal'] = 1

    return signals

#%%
# Functions from volswitch.py
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
    signals['vols_switch_buy_signal'] = 0
    signals['vols_switch_sell_signal'] = 0
    
    # Create signals
    signals.loc[(stock_df['Buy_Signal'] == True), 'vols_switch_buy_signal'] = 1
    signals.loc[(stock_df['Sell_Signal'] == True), 'vols_switch_sell_signal'] = 1
 
    return signals

#%%
# Functions from voltyexpancloselx.py
import talib
from talib import MA_Type

def volty_expan_close_lx(df, num_atrs=2, length=14, ma_type='SMA'):
    df = df.copy()  # Avoid modifying the original DataFrame

    # Compute ATR
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=length)

    # Map string input to TALib's MA_Type
    ma_type_mapping = {
        'SMA': MA_Type.SMA, 'EMA': MA_Type.EMA, 'WMA': MA_Type.WMA,
        'DEMA': MA_Type.DEMA, 'TEMA': MA_Type.TEMA, 'TRIMA': MA_Type.TRIMA,
        'KAMA': MA_Type.KAMA, 'MAMA': MA_Type.MAMA
    }
    ma_type = ma_type_mapping.get(ma_type, MA_Type.SMA)

    # Compute ATR moving average
    df['ATR_MA'] = talib.MA(df['ATR'], timeperiod=length, matype=ma_type)

    # Initialize signal column
    df['volty_expan_close_lx_sell_signal'] = 0

    # Avoid looping; use vectorized calculation
    condition = (df['Low'] < (df['Close'].shift(1) - num_atrs * df['ATR_MA'])) & \
                (df['Open'] < (df['Close'].shift(1) - num_atrs * df['ATR_MA']))
    
    df.loc[condition, 'volty_expan_close_lx_sell_signal'] = 1

    # Drop intermediate columns if not needed
    df.drop(columns=['ATR', 'ATR_MA'], inplace=True)

    return df[['volty_expan_close_lx_sell_signal']]

#%%
# Functions from vpnstrat.py
def vpn_signals(df, length=14, ema_length=14, average_length=10, rsi_length=14, volume_average_length=50, highest_length=14, atr_length=14, factor=1, critical_value=10, rsi_max_overbought_level=90, num_atrs=1, average_type='simple'):
    vwap = volume.VolumeWeightedAveragePrice(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=average_length)

    # Calculate VPN 
    vpn = (df['Volume'] * factor * np.where(df['Close'].diff() > 0, 1, -1)).ewm(span=length).mean() / df['Volume'].ewm(span=length).mean()
    vpn_average = vpn.ewm(span=average_length).mean() if average_type == 'exponential' else vpn.rolling(window=average_length).mean()
    
    # Create signal column
    signals = pd.DataFrame(index=df.index)
    signals['vpn_buy_signal'] = 0
    signals['vpn_sell_signal'] = 0

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
    signals.loc[buy_condition, 'vpn_buy_signal'] = 1
    signals.loc[sell_condition, 'vpn_sell_signal'] = 1

    return signals

#%%
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
    df['vwma_breakouts_buy_signal'] = 0
    df['vwma_breakouts_sell_signal'] = 0
    df.loc[df['VWMA'] > df['SMA'], 'vwma_breakouts_buy_signal'] = 1
    df.loc[df['VWMA'] < df['SMA'], 'vwma_breakouts_sell_signal'] = 1

    return df[['vwma_breakouts_buy_signal', 'vwma_breakouts_sell_signal']]

#%%
# PPO (Percentage Price Oscillator)
def ppo_signals(stock_data, fast_window=12, slow_window=26, signal_window=9):
    """
    Computes PPO crossover signals.

    Returns:
    A DataFrame with 'PPO_signal'.
    """
    signals = pd.DataFrame(index=stock_data.index)

    # Calculate PPO and PPO signal
    ppo = momentum.PercentagePriceOscillator(stock_data['Close'], fast_window, slow_window, signal_window)
    signals['PPO'] = ppo.ppo()
    signals['PPO_Signal'] = ppo.ppo_signal()

    # Generate buy/sell signals on PPO crossover
    signals['PPO_buy_signal'] = 0
    signals['PPO_sell_signal'] = 0
    signals.loc[(signals['PPO'] > signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) <= signals['PPO_Signal'].shift(1)), 'PPO_buy_signal'] = 1
    
    signals.loc[(signals['PPO'] < signals['PPO_Signal']) & 
                (signals['PPO'].shift(1) >= signals['PPO_Signal'].shift(1)), 'PPO_sell_signal'] = 1

    return signals

#%% 
# Awesome Oscillator Zero Cross Strategy
def Awesome_Oscillator_signals(stock_df):
    """
    Computes Awesome Oscillator zero-crossing signals.

    Returns:
    A DataFrame with 'Ao_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Awesome Oscillator
    ao_indicator = momentum.AwesomeOscillatorIndicator(high=stock_df['High'], low=stock_df['Low'])
    signals['AO'] = ao_indicator.awesome_oscillator()

    # Generate signals on zero-crossing
    signals['Ao_buy_signal'] = 0
    signals['Ao_sell_signal'] = 0
    signals.loc[(signals['AO'].shift(1) < 0) & (signals['AO'] >= 0), 'Ao_buy_signal'] = 1
    signals.loc[(signals['AO'].shift(1) >= 0) & (signals['AO'] < 0), 'Ao_sell_signal'] = 1

    return signals

#%% 
# KAMA Cross Strategy
def kama_cross_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes KAMA crossover and price cross signals.

    Returns:
    A DataFrame with 'kama_cross_signal' and 'kama_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Fast and Slow KAMA
    fast_kama = momentum.kama(stock_df['Close'], window=fast_period)
    slow_kama = momentum.kama(stock_df['Close'], window=slow_period)

    # Generate crossover signals
    signals['kama_cross_buy_signal'] = 0
    signals['kama_cross_sell_signal'] = 0
    signals.loc[(fast_kama > slow_kama) & (fast_kama.shift(1) <= slow_kama.shift(1)) & (stock_df['Close'] > fast_kama), 
                'kama_cross_buy_signal'] = 1
    
    signals.loc[(fast_kama < slow_kama) & (fast_kama.shift(1) >= slow_kama.shift(1)) & (stock_df['Close'] < fast_kama), 
                'kama_cross_sell_signal'] = 1

    signals.loc[(stock_df['Close'] > fast_kama) & (stock_df['Close'].shift(1) <= fast_kama.shift(1)), 'kama_cross_buy_signal'] = 1
    signals.loc[(stock_df['Close'] < fast_kama) & (stock_df['Close'].shift(1) >= fast_kama.shift(1)), 'kama_cross_sell_signal'] = 1

    return signals

#%% 
# Stochastic Oscillator Strategy
def stoch_signals(stock_df, fast_period=10, slow_period=20):
    """
    Computes Stochastic Oscillator crossover signals.

    Returns:
    A DataFrame with 'stoch_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Stochastic Oscillator
    stoch = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=14, smooth_window=3)
    signals['%K'] = stoch.stoch()
    signals['%D'] = stoch.stoch_signal()

    # Generate crossover signals
    signals['stoch_buy_signal'] = 0
    signals['stoch_sell_signal'] = 0
    signals.loc[
        (signals['%K'] > signals['%D']) & (signals['%K'].shift(1) <= signals['%D'].shift(1)), 'stoch_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['%K'] < signals['%D']) & (signals['%K'].shift(1) >= signals['%D'].shift(1)), 'stoch_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['%K', '%D'], axis=1, inplace=True)

    return signals

#%% 
# True Strength Index (TSI) Strategy
def tsi_signals(stock_df, window_slow=25, window_fast=13):
    """
    Computes True Strength Index (TSI) crossover signals.

    Returns:
    A DataFrame with 'tsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate TSI
    tsi = momentum.TSIIndicator(stock_df['Close'], window_slow, window_fast)
    signals['TSI'] = tsi.tsi()

    # Generate signals based on TSI zero-crossing
    signals['tsi_buy_signal'] = 0
    signals['tsi_sell_signal'] = 0
    signals.loc[(signals['TSI'] > 0) & (signals['TSI'].shift(1) <= 0), 'tsi_buy_signal'] = 1
    signals.loc[(signals['TSI'] < 0) & (signals['TSI'].shift(1) >= 0), 'tsi_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['TSI'], axis=1, inplace=True)

    return signals

#%% 
# Williams %R Overbought/Oversold Strategy
def williams_signals(stock_df, lbp=14):
    """
    Computes Williams %R overbought and oversold signals.

    Returns:
    A DataFrame with 'williams_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Williams %R
    williams_r = momentum.WilliamsRIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], lbp)
    signals['WilliamsR'] = williams_r.williams_r()

    # Generate overbought/oversold signals
    signals['williams_buy_signal'] = 0
    signals['williams_sell_signal'] = 0
    signals.loc[signals['WilliamsR'] <= -80, 'williams_buy_signal'] = 1
    signals.loc[signals['WilliamsR'] >= -20, 'williams_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['WilliamsR'], axis=1, inplace=True)

    return signals

#%% 
# Rate of Change (ROC) Overbought/Oversold Strategy
def roc_signals(stock_df, window=12):
    """
    Computes ROC overbought and oversold signals.

    Returns:
    A DataFrame with 'roc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Rate of Change (ROC)
    roc = momentum.ROCIndicator(stock_df['Close'], window)
    signals['ROC'] = roc.roc()

    # Generate overbought/oversold signals
    signals['roc_overbought_signal'] = 0
    signals['roc_oversold_signal'] = 0
    signals.loc[signals['ROC'] >= 10, 'roc_overbought_signal'] = 1
    signals.loc[signals['ROC'] <= -10, 'roc_oversold_signal'] = 1

    # Drop temporary column
    signals.drop(['ROC'], axis=1, inplace=True)

    return signals

#%% 
# Relative Strength Index (RSI) Overbought/Oversold Strategy
def rsi_signals(stock_df, window=14):
    """
    Computes RSI overbought and oversold signals.

    Returns:
    A DataFrame with 'rsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi = momentum.RSIIndicator(stock_df['Close'], window)
    signals['RSI'] = rsi.rsi()

    # Generate overbought/oversold signals
    signals['rsi_overbought_signal'] = 0
    signals['rsi_oversold_signal'] = 0
    signals.loc[signals['RSI'] >= 70, 'rsi_overbought_signal'] = 1
    signals.loc[signals['RSI'] <= 30, 'rsi_oversold_signal'] = 1

    # Drop temporary column
    signals.drop(['RSI'], axis=1, inplace=True)

    return signals

#%% 
# Stochastic RSI Overbought/Oversold Strategy
def stochrsi_signals(stock_df, window=14, smooth1=3, smooth2=3):
    """
    Computes StochRSI overbought and oversold signals.

    Returns:
    A DataFrame with 'stochrsi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate StochRSI
    stoch_rsi = momentum.StochRSIIndicator(stock_df['Close'], window, smooth1, smooth2)
    signals['StochRSI'] = stoch_rsi.stochrsi()

    # Generate overbought/oversold signals
    signals['stochrsi_overbought_signal'] = 0
    signals['stochrsi_oversold_signal'] = 0
    signals.loc[signals['StochRSI'] >= 0.8, 'stochrsi_overbought_signal'] = 1
    signals.loc[signals['StochRSI'] <= 0.2, 'stochrsi_oversold_signal'] = 1

    # Drop temporary column
    signals.drop(['StochRSI'], axis=1, inplace=True)

    return signals

#%% 
# Commodity Channel Index (CCI) Strategy
def cci_signals(stock_df, window=20, constant=0.015, overbought=100, oversold=-100):
    """
    Computes CCI trend direction and trading signals.

    Returns:
    A DataFrame with 'cci_direction' and 'cci_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create CCI Indicator
    cci = trend.CCIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window, constant)
    signals['CCI'] = cci.cci()

    # Determine market direction
    signals['cci_bullish_signal'] = 0
    signals['cci_bearish_signal'] = 0
    signals.loc[signals['CCI'] > oversold, 'cci_bullish_signal'] = 1
    signals.loc[signals['CCI'] < overbought, 'cci_bearish_signal'] = 1

    # Generate buy/sell signals based on overbought/oversold conditions
    signals['cci_buy_signal'] = 0
    signals['cci_sell_signal'] = 0
    signals.loc[(signals['CCI'] > overbought) & (signals['CCI'].shift(1) <= overbought), 'cci_buy_signal'] = 1
    signals.loc[(signals['CCI'] < oversold) & (signals['CCI'].shift(1) >= oversold), 'cci_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['CCI'], axis=1, inplace=True)

    return signals

#%% 
# Detrended Price Oscillator (DPO) Strategy
def dpo_signals(stock_df, window=20):
    """
    Computes DPO trend direction and crossover signals.

    Returns:
    A DataFrame with 'dpo_direction_Signal' and 'dpo_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create DPO Indicator
    dpo = trend.DPOIndicator(stock_df['Close'], window)
    signals['DPO'] = dpo.dpo()

    # Determine market direction
    signals['dpo_overbought_signal'] = 0
    signals['dpo_oversold_signal'] = 0
    signals.loc[signals['DPO'] > 0, 'dpo_overbought_signal'] = 1
    signals.loc[signals['DPO'] < 0, 'dpo_oversold_signal'] = 1

    # Generate buy/sell signals based on zero-crossing
    signals['dpo_buy_signal'] = 0
    signals['dpo_sell_signal'] = 0
    signals.loc[(signals['DPO'] > 0) & (signals['DPO'].shift(1) <= 0), 'dpo_buy_signal'] = 1
    signals.loc[(signals['DPO'] < 0) & (signals['DPO'].shift(1) >= 0), 'dpo_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['DPO'], axis=1, inplace=True)

    return signals

#%% 
# Exponential Moving Average (EMA) Crossover Strategy
def ema_signals(stock_df, short_window=12, long_window=26):
    """
    Computes EMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'EMA_Direction_Signal' and 'EMA_Signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate short-term and long-term EMAs
    ema_short = stock_df['Close'].ewm(span=short_window, adjust=False).mean()
    ema_long = stock_df['Close'].ewm(span=long_window, adjust=False).mean()

    # Determine market direction
    signals['EMA_bullish_signal'] = 0
    signals['EMA_bearish_signal'] = 0
    signals.loc[ema_short > ema_long, 'EMA_bullish_signal'] = 1
    signals.loc[ema_short < ema_long, 'EMA_bearish_signal'] = 1

    # Generate buy/sell signals based on EMA crossovers
    signals['EMA_buy_signal'] = 0
    signals['EMA_sell_signal'] = 0
    signals.loc[(ema_short > ema_long) & (ema_short.shift(1) <= ema_long.shift(1)), 'EMA_buy_signal'] = 1
    signals.loc[(ema_short < ema_long) & (ema_short.shift(1) >= ema_long.shift(1)), 'EMA_sell_signal'] = 1

    return signals

#%% 
# Ichimoku Cloud Strategy
def ichimoku_signals(stock_df, window1=9, window2=26):
    """
    Computes Ichimoku trend direction and crossover signals.

    Returns:
    A DataFrame with 'ichi_signal' and 'ichi_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create Ichimoku Indicator
    ichimoku = trend.IchimokuIndicator(stock_df['High'], stock_df['Low'], window1, window2)
    signals['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
    signals['kijun_sen'] = ichimoku.ichimoku_base_line()
    signals['senkou_span_a'] = ichimoku.ichimoku_a()
    signals['senkou_span_b'] = ichimoku.ichimoku_b()

    # Generate crossover signals
    signals['ichi_buy_signal'] = 0
    signals['ichi_sell_signal'] = 0
    signals.loc[(signals['tenkan_sen'] > signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) <= signals['kijun_sen'].shift(1)), 'ichi_buy_signal'] = 1
    
    signals.loc[(signals['tenkan_sen'] < signals['kijun_sen']) & 
                (signals['tenkan_sen'].shift(1) >= signals['kijun_sen'].shift(1)), 'ichi_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b'], axis=1, inplace=True)

    return signals

#%% 
# Know Sure Thing (KST) Strategy
def kst_signals(stock_df, roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15, nsig=9):
    """
    Computes KST trend direction and crossover signals.

    Returns:
    A DataFrame with 'kst_signal' and 'kst_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create KST Indicator
    kst = trend.KSTIndicator(stock_df['Close'], roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig)
    signals['KST'] = kst.kst()
    signals['KST_Signal'] = kst.kst_sig()

    # Generate crossover signals
    signals['kst_buy_signal'] = 0
    signals['kst_sell_signal'] = 0
    signals.loc[(signals['KST'] > signals['KST_Signal']) & 
                (signals['KST'].shift(1) <= signals['KST_Signal'].shift(1)), 'kst_buy_signal'] = 1
    
    signals.loc[(signals['KST'] < signals['KST_Signal']) & 
                (signals['KST'].shift(1) >= signals['KST_Signal'].shift(1)), 'kst_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['KST', 'KST_Signal'], axis=1, inplace=True)

    return signals

#%% 
# Moving Average Convergence Divergence (MACD) Strategy
def macd_signals(stock_df, window_slow=26, window_fast=12, window_sign=9):
    """
    Computes MACD trend direction and crossover signals.

    Returns:
    A DataFrame with 'macd_signal' and 'macd_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Create MACD Indicator
    macd = trend.MACD(stock_df['Close'], window_slow, window_fast, window_sign)
    signals['MACD'] = macd.macd()
    signals['MACD_Signal'] = macd.macd_signal()

    # Generate crossover signals
    signals['macd_conv_buy_signal'] = 0
    signals['macd_conv_sell_signal'] = 0
    signals.loc[(signals['MACD'] > signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) <= signals['MACD_Signal'].shift(1)), 'macd_conv_buy_signal'] = 1
    
    signals.loc[(signals['MACD'] < signals['MACD_Signal']) & 
                (signals['MACD'].shift(1) >= signals['MACD_Signal'].shift(1)), 'macd_conv_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['MACD', 'MACD_Signal'], axis=1, inplace=True)

    return signals

#%% 
# Golden Cross SMA Strategy
def golden_ma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes SMA crossover trend and trading signals.

    Returns:
    A DataFrame with 'ma_direction' and 'ma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long SMAs
    short_sma = trend.SMAIndicator(stock_df['Close'], short_period).sma_indicator()
    long_sma = trend.SMAIndicator(stock_df['Close'], long_period).sma_indicator()

    # Generate crossover signals
    signals['ma_buy_signal'] = 0
    signals['ma_sell_signal'] = 0
    signals.loc[(short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1)), 'ma_buy_signal'] = 1
    signals.loc[(short_sma <= long_sma) & (short_sma.shift(1) > long_sma.shift(1)), 'ma_sell_signal'] = 1

    return signals

#%% 
# 5-8-13 SMA Strategy
def strategy_5_8_13(stock_df):
    """
    Computes 5-8-13 SMA crossover trend direction.

    Returns:
    A DataFrame with '5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short SMAs
    sma5 = trend.SMAIndicator(stock_df['Close'], 5).sma_indicator()
    sma8 = trend.SMAIndicator(stock_df['Close'], 8).sma_indicator()
    sma13 = trend.SMAIndicator(stock_df['Close'], 13).sma_indicator()

    # Determine market direction
    signals['5_8_13_buy_signal'] = 0
    signals['5_8_13_sell_signal'] = 0
    signals.loc[(sma5 > sma8) & (sma8 > sma13), '5_8_13_buy_signal'] = 1
    signals.loc[(sma5 < sma8) & (sma8 < sma13), '5_8_13_sell_signal'] = 1

    return signals

#%% 
# 5-8-13 WMA Strategy
def strategy_w5_8_13(stock_df):
    """
    Computes 5-8-13 WMA crossover trend direction.

    Returns:
    A DataFrame with 'w5_8_13_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute weighted moving averages
    wma5 = trend.WMAIndicator(stock_df['Close'], 5).wma()
    wma8 = trend.WMAIndicator(stock_df['Close'], 8).wma()
    wma13 = trend.WMAIndicator(stock_df['Close'], 13).wma()

    # Determine market direction
    signals['w5_8_13_buy_signal'] = 0
    signals['w5_8_13_sell_signal'] = 0
    signals.loc[(wma5 > wma8) & (wma8 > wma13), 'w5_8_13_buy_signal'] = 1
    signals.loc[(wma5 < wma8) & (wma8 < wma13), 'w5_8_13_sell_signal'] = 1

    return signals

#%% 
# Keltner Channel Strategy
def keltner_channel_strategy(stock_df, window=20, window_atr=10, multiplier=2):
    """
    Computes Keltner Channel breakout signals.

    Returns:
    A DataFrame with 'kc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Keltner Channel bands
    keltner_channel = volatility.KeltnerChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window, window_atr, multiplier)
    signals['kc_upper'] = keltner_channel.keltner_channel_hband()
    signals['kc_lower'] = keltner_channel.keltner_channel_lband()

    # Generate breakout signals
    signals['kc_buy_signal'] = 0
    signals['kc_sell_signal'] = 0
    signals.loc[stock_df['Close'] > signals['kc_upper'], 'kc_buy_signal'] = 1
    signals.loc[stock_df['Close'] < signals['kc_lower'], 'kc_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['kc_upper', 'kc_lower'], axis=1, inplace=True)

    return signals

#%% 
# Chaikin Money Flow (CMF) Strategy
def cmf_signals(stock_df, window=20, threshold=0.1):
    """
    Computes CMF trend signals.

    Returns:
    A DataFrame with 'cmf_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute CMF
    cmf = volume.ChaikinMoneyFlowIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['CMF'] = cmf.chaikin_money_flow()

    # Generate signals
    signals['cmf_buy_signal'] = 0
    signals['cmf_sell_signal'] = 0
    signals.loc[signals['CMF'] > threshold, 'cmf_buy_signal'] = 1
    signals.loc[signals['CMF'] < -threshold, 'cmf_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['CMF'], axis=1, inplace=True)

    return signals

#%% 
# Money Flow Index (MFI) Strategy
def mfi_signals(stock_df, window=14):
    """
    Computes MFI overbought/oversold signals.

    Returns:
    A DataFrame with 'mfi_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute MFI
    mfi = volume.MFIIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], stock_df['Volume'], window)
    signals['MFI'] = mfi.money_flow_index()

    # Generate signals
    signals['mfi_buy_signal'] = 0
    signals['mfi_sell_signal'] = 0
    signals.loc[signals['MFI'] > 80, 'mfi_buy_signal'] = 1
    signals.loc[signals['MFI'] < 20, 'mfi_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['MFI'], axis=1, inplace=True)

    return signals

#%% 
# Ease of Movement (EOM) Strategy
def eom_signals(stock_df, window=14):
    """
    Computes EOM trend signals.

    Returns:
    A DataFrame with 'eom_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute EOM
    eom = volume.EaseOfMovementIndicator(stock_df['High'], stock_df['Low'], stock_df['Volume'], window)
    signals['EOM'] = eom.ease_of_movement()

    # Generate signals
    signals['eom_buy_signal'] = 0
    signals['eom_sell_signal'] = 0
    signals.loc[signals['EOM'] > 0, 'eom_buy_signal'] = 1
    signals.loc[signals['EOM'] < 0, 'eom_sell_signal'] = 1

    # Drop temporary column
    signals.drop(['EOM'], axis=1, inplace=True)

    return signals

#%%
# Aroon Strategy
def aroon_strategy(stock_df, window=25):
    """
    Computes Aroon trend strength and direction.

    Returns:
    A DataFrame with 'aroon_Trend_Strength', 'aroon_direction_signal', and 'aroon_signal'.
    """
    # Create Aroon Indicator
    aroon = trend.AroonIndicator(stock_df['Close'], stock_df['Low'], window=window)
    
    # Create a DataFrame to store signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Aroon_Up'] = aroon.aroon_up()
    signals['Aroon_Down'] = aroon.aroon_down()

    # Determine trend strength
    signals['aroon_Trend_signal'] = 'weak'
    signals.loc[(signals['Aroon_Up'] >= 70) | (signals['Aroon_Down'] >= 70), 'aroon_Trend_Strength'] = 'strong'

    # Determine direction signal
    signals['aroon_direction_signal'] = 'bearish'
    signals.loc[signals['Aroon_Up'] > signals['Aroon_Down'], 'aroon_direction_signal'] = 'bullish'

    # Generate trading signal
    signals['aroon_buy_signal'] = 0
    signals['aroon_sell_signal'] = 0
    signals.loc[
        (signals['aroon_direction_signal'] == 'bullish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bearish'), 'aroon_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['aroon_direction_signal'] == 'bearish') & 
        (signals['aroon_direction_signal'].shift(1) == 'bullish'), 'aroon_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['Aroon_Up', 'Aroon_Down', 'aroon_Trend_signal', 'aroon_direction_signal'], axis=1, inplace=True)

    return signals

#%%
#RSI Divergence strategy
def rsi_signals_with_divergence(stock_df, window=14, long_threshold=30, short_threshold=70, width=10):
    """
    Computes RSI signals with divergence detection.

    Returns:
    A DataFrame with 'RSI_signal' and 'RSI' values.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate RSI
    rsi_indicator = momentum.RSIIndicator(stock_df['Close'], window=window)
    signals['RSI'] = rsi_indicator.rsi()

    # Generate RSI overbought/oversold signals
    signals['RSI_buy_signal'] = 0
    signals['RSI_sell_signal'] = 0
    signals.loc[signals['RSI'] < long_threshold, 'RSI_buy_signal'] = 1
    signals.loc[signals['RSI'] > short_threshold, 'RSI_sell_signal'] = 1

    return signals

# %%
# Mass Index
def mass_index_signals(stock_df, fast_window=9, slow_window=25):
    """
    Computes Mass Index reversal signals.

    Returns:
    A DataFrame with 'Mass_Index' and 'mass_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Mass Index
    mass_index = trend.MassIndex(stock_df['High'], stock_df['Low'], window_fast=fast_window, window_slow=slow_window)
    signals['Mass_Index'] = mass_index.mass_index()

    # Calculate Short & Long EMAs for trend direction
    short_ema = trend.EMAIndicator(stock_df['Close'], window=fast_window).ema_indicator()
    long_ema = trend.EMAIndicator(stock_df['Close'], window=slow_window).ema_indicator()

    # Set thresholds for Mass Index reversals
    reversal_bulge_threshold = 27
    reversal_bulge_exit_threshold = 26.5

    # Determine if the market is in a downtrend
    in_downtrend = short_ema < long_ema

    # Generate signals for reversals
    signals['mass_buy_signal'] = 0
    signals['mass_sell_signal'] = 0
    signals.loc[(in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_buy_signal'] = 1
    
    signals.loc[(~in_downtrend) & (signals['Mass_Index'] > reversal_bulge_threshold) & 
                (signals['Mass_Index'].shift(1) <= reversal_bulge_exit_threshold), 'mass_sell_signal'] = 1

    return signals
#%%
# PSAR 
def psar_signals(stock_df, step=0.02, max_step=0.2):
    """
    Computes Parabolic SAR trend direction and signals.

    Returns:
    A DataFrame with 'psar_direction' and 'psar_signal' columns.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute PSAR values
    psar_indicator = trend.PSARIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], step=step, max_step=max_step)
    psar_values = psar_indicator.psar()

    # Generate buy/sell signals based on crossovers
    signals['psar_buy_signal'] = 0
    signals['psar_sell_signal'] = 0
    signals.loc[(stock_df['Close'] > psar_values) & (stock_df['Close'].shift(1) <= psar_values.shift(1)), 'psar_buy_signal'] = 1
    signals.loc[(stock_df['Close'] < psar_values) & (stock_df['Close'].shift(1) >= psar_values.shift(1)), 'psar_sell_signal'] = 1

    return signals

#%%
# STC
def stc_signals(stock_df, window_slow=50, window_fast=23, cycle=10, smooth1=3, smooth2=3):
    """
    Computes Schaff Trend Cycle (STC) signals.

    Returns:
    A DataFrame with 'stc_signal' and 'stc_direction'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate STC
    stc_indicator = trend.STCIndicator(stock_df['Close'], window_slow, window_fast, cycle, smooth1, smooth2)
    signals['STC'] = stc_indicator.stc()

    # Determine overbought/oversold conditions
    signals['stc_overbought_signal'] = 0
    signals['stc_oversold_signal'] = 0
    signals.loc[signals['STC'] > 75, 'stc_overbought_signal'] = 1
    signals.loc[signals['STC'] < 25, 'stc_oversold_signal'] = 1

    return signals

#%%
# Vortex
def vortex_signals(stock_df, window=14):
    """
    Computes Vortex Indicator trend direction and signals.

    Returns:
    A DataFrame with 'vortex_signal' and 'vortex_direction_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute Vortex Indicator values
    vortex = trend.VortexIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Positive'] = vortex.vortex_indicator_pos()
    signals['Negative'] = vortex.vortex_indicator_neg()

    # Generate trading signals based on crossovers
    signals['vortex_buy_signal'] = 0
    signals['vortex_sell_signal'] = 0
    signals.loc[
        (signals['Positive'] > signals['Negative']) & 
        (signals['Positive'].shift(1) <= signals['Negative'].shift(1)), 'vortex_buy_signal'
    ] = 1
    
    signals.loc[
        (signals['Positive'] < signals['Negative']) & 
        (signals['Positive'].shift(1) >= signals['Negative'].shift(1)), 'vortex_sell_signal'
    ] = 1

    # Drop temporary columns
    signals.drop(['Positive', 'Negative'], axis=1, inplace=True)

    return signals
#%%
# Golden Cross WMA
def golden_wma_signals(stock_df, short_period=50, long_period=200):
    """
    Computes Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()


    # Generate crossover signals
    signals['wma_buy_signal'] = 0
    signals['wma_sell_signal'] = 0
    signals.loc[
        (short_wma > long_wma) & 
        (short_wma.shift(1) <= long_wma.shift(1)), 'wma_buy_signal'
    ] = 1
    
    signals.loc[
        (short_wma <= long_wma) & 
        (short_wma.shift(1) > long_wma.shift(1)), 'wma_sell_signal'
    ] = 1

    return signals
#%%
# 13-26 MA STrategy
def short_wma_signals(stock_df, short_period=13, long_period=26):
    """
    Computes 13-26 Weighted Moving Average (WMA) crossover signals.

    Returns:
    A DataFrame with 'wma_direction' and 'wma_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Compute short and long WMAs
    short_wma = trend.WMAIndicator(stock_df['Close'], short_period).wma()
    long_wma = trend.WMAIndicator(stock_df['Close'], long_period).wma()

    # Generate crossover signals
    signals['wma_buy_signal'] = 0
    signals['wma_sell_signal'] = 0
    signals.loc[
        (short_wma > long_wma) & (short_wma.shift(1) <= long_wma.shift(1)), 'wma_buy_signal'
    ] = 1
    
    signals.loc[
        (short_wma <= long_wma) & (short_wma.shift(1) > long_wma.shift(1)), 'wma_sell_signal'
    ] = 1

    return signals

# %%
# Donchain Channel
def donchian_channel_strategy(stock_df, window=20):
    """
    Computes Donchian Channel breakout signals.

    Returns:
    A DataFrame with 'dc_signal'.
    """
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate Donchian Channel
    donchian = volatility.DonchianChannel(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['Upper_Channel'] = donchian.donchian_channel_hband()
    signals['Lower_Channel'] = donchian.donchian_channel_lband()

    # Determine long and short signals (breakout strategy)
    signals['dc_buy_signal'] = 0
    signals['dc_sell_signal'] = 0
    signals.loc[stock_df['Close'] > signals['Upper_Channel'], 'dc_buy_signal'] = 1
    signals.loc[stock_df['Close'] < signals['Lower_Channel'], 'dc_sell_signal'] = 1

    # Drop temporary columns
    signals.drop(['Upper_Channel', 'Lower_Channel'], axis=1, inplace=True)

    return signals

#%% 
# Iron Bot
def ironbot_trend_filter(stock_df, z_length=40, analysis_window=44, high_trend_limit=23.6, 
                         low_trend_limit=78.6, use_ema=False, ema_length=200):
    """
    Implements the IronBot Statistical Trend Filter strategy.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close' prices.
    - z_length: Window for Z-score calculation.
    - analysis_window: Trend analysis period.
    - high_trend_limit, low_trend_limit: Fibonacci levels for trend confirmation.
    - use_ema: Whether to use an EMA filter.
    - ema_length: EMA period.

    Returns:
    - A DataFrame with trend levels and trading signals.
    """
    
    signals = pd.DataFrame(index=stock_df.index)

    # === Compute Z-Score ===
    stock_df['z_score'] = (stock_df['Close'] - stock_df['Close'].rolling(z_length).mean()) / stock_df['Close'].rolling(z_length).std()

    # === Compute Fibonacci Trend Levels ===
    stock_df['highest_high'] = stock_df['High'].rolling(analysis_window).max()
    stock_df['lowest_low'] = stock_df['Low'].rolling(analysis_window).min()
    stock_df['price_range'] = stock_df['highest_high'] - stock_df['lowest_low']

    stock_df['high_trend_level'] = stock_df['highest_high'] - stock_df['price_range'] * (high_trend_limit / 100)
    stock_df['trend_line'] = stock_df['highest_high'] - stock_df['price_range'] * 0.5
    stock_df['low_trend_level'] = stock_df['highest_high'] - stock_df['price_range'] * (low_trend_limit / 100)

    # === Compute EMA Filter ===
    if use_ema:
        ema = trend.EMAIndicator(stock_df['Close'], ema_length)
        stock_df['ema'] = ema.ema_indicator()
        stock_df['ema_bullish'] = stock_df['Close'] >= stock_df['ema']
        stock_df['ema_bearish'] = stock_df['Close'] <= stock_df['ema']
    else:
        stock_df['ema_bullish'] = True
        stock_df['ema_bearish'] = True

    # === Entry Conditions ===
    stock_df['can_long'] = (stock_df['Close'] >= stock_df['trend_line']) & (stock_df['Close'] >= stock_df['high_trend_level']) & (stock_df['z_score'] >= 0) & stock_df['ema_bullish']
    stock_df['can_short'] = (stock_df['Close'] <= stock_df['trend_line']) & (stock_df['Close'] <= stock_df['low_trend_level']) & (stock_df['z_score'] <= 0) & stock_df['ema_bearish']

    # === Generate Trading Signals ===
    signals['ironbot_buy_signal'] = 0
    signals['ironbot_sell_signal'] = 0
    signals.loc[stock_df['can_long'], 'ironbot_buy_signal'] = 1
    signals.loc[stock_df['can_short'], 'ironbot_sell_signal'] = 1

    return signals

#%%
# High Volume Points
def high_volume_points(stock_df, left_bars=15, filter_vol=2.0, lookback_period=300):
    """
    Identifies high-volume pivot points in price data.

    Parameters:
    - stock_df: DataFrame containing 'High', 'Low', 'Close', 'Volume'.
    - left_bars: Number of bars before & after a pivot high/low for validation.
    - filter_vol: Minimum volume threshold to filter significant pivot points.
    - lookback_period: Number of bars to consider for percentile volume ranking.

    Returns:
    - A DataFrame with high-volume pivot points and liquidity grab signals.
    """

    signals = pd.DataFrame(index=stock_df.index)

    # === Compute Pivot Highs & Lows ===
    stock_df['pivot_high'] = stock_df['High'][(stock_df['High'] == stock_df['High'].rolling(window=left_bars*2+1, center=True).max())]
    stock_df['pivot_low'] = stock_df['Low'][(stock_df['Low'] == stock_df['Low'].rolling(window=left_bars*2+1, center=True).min())]

    # === Compute Normalized Volume ===
    stock_df['rolling_volume'] = stock_df['Volume'].rolling(window=left_bars * 2).sum()
    reference_vol = stock_df['rolling_volume'].rolling(lookback_period).quantile(0.95)
    stock_df['norm_vol'] = stock_df['rolling_volume'] / reference_vol * 5  # Normalize between 0-6 scale

    # === Apply Volume Filter ===
    stock_df['valid_pivot_high'] = (stock_df['pivot_high'].notna()) & (stock_df['norm_vol'] > filter_vol)
    stock_df['valid_pivot_low'] = (stock_df['pivot_low'].notna()) & (stock_df['norm_vol'] > filter_vol)

    # === Detect Liquidity Grabs ===
    stock_df['liquidity_grab_high'] = stock_df['valid_pivot_high'] & (stock_df['Close'] < stock_df['pivot_high'].shift(1))
    stock_df['liquidity_grab_low'] = stock_df['valid_pivot_low'] & (stock_df['Close'] > stock_df['pivot_low'].shift(1))

    # === Store Relevant Features for ML Training ===
    signals['high_volume_pivot'] = stock_df['valid_pivot_high'] | stock_df['valid_pivot_low']
    signals['liquidity_grab'] = stock_df['liquidity_grab_high'] | stock_df['liquidity_grab_low']

    return signals

#%%
# Day of the week
def day_of_the_week(df):
    # Make a copy of the dataframe to avoid modifying the original data
    df_copy = df.copy()
    
    # Check if the index is a datetime index
    if isinstance(df_copy.index, pd.DatetimeIndex):
        # Pandas dayofweek: Monday=0, Sunday=6; add 1 to map to 1-7
        day_numbers = df_copy.index.dayofweek + 1
    # Alternatively, if you have a 'Date' column:
    elif 'Date' in df_copy.columns:
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        day_numbers = df_copy['Date'].dt.dayofweek + 1
    else:
        raise ValueError("No datetime index or 'Date' column found in the DataFrame.")
    
    # Create a new DataFrame with the computed day of week numbers
    days_df = pd.DataFrame({'day_of_week': day_numbers}, index=df_copy.index)
    return days_df

