import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.api as sm

def calculate_RSquared(df, length=18):
    if len(df) < length:
        return np.nan  # Prevent errors when there isn't enough data

    X = np.arange(length)
    X = sm.add_constant(X) 
    Y = df.tail(length).values
    model = sm.OLS(Y, X)
    results = model.fit()
    return results.rsquared

def calculate_slope(df, length=18):
    if len(df) < length:
        return np.nan  # Prevent errors when there isn't enough data

    X = np.arange(length)
    Y = df.tail(length).values
    return linregress(X, Y).slope

def calculate_moving_average(df, length=18, average_type='simple'):
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
    signals['RSquared'] = df['Close'].rolling(length).apply(calculate_RSquared, raw=False)
    signals['slope'] = df['Close'].rolling(length).apply(calculate_slope, raw=False)

    # Apply moving average calculation safely
    signals['ma'] = df['Close'].rolling(average_length).apply(lambda x: calculate_moving_average(pd.Series(x), average_length, average_type), raw=False)

    # Fill missing values in ma to prevent NaN comparison errors
    signals['ma'].fillna(method='ffill', inplace=True)

    signals['RSquared_lag'] = signals['RSquared'].shift(lag)
    signals['r2trend_signals'] = 0

    # Strong Uptrend Condition
    cond1 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] > lr_crit_level) & \
            (df['Close'] > signals['ma'])  # Safe comparison

    signals.loc[cond1, 'r2trend_signals'] = 1

    # Strong Downtrend Condition
    cond2 = (signals['RSquared'] > max_level) & \
            (signals['RSquared'] > signals['RSquared_lag']) & \
            (signals['slope'] < -lr_crit_level) & \
            (df['Close'] < signals['ma'])  # Safe comparison

    signals.loc[cond2, 'r2trend_signals'] = -1
    signals.drop(['slope', 'ma', 'RSquared_lag'], axis=1, inplace=True)

    return signals

