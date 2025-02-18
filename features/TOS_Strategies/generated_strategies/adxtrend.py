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
    signals['adx_trend_signal'] = np.where(
        (
            ((signals['ADX'] > signals['min_ADX'] * mult) & (signals['ADX'] > crit_level)) |
            ((signals['ADX'] > trend_level) & (signals['ADX'] < max_level))
        ) & (close > signals['avg_price']), 'buy', 
        np.where(
            (
                ((signals['ADX'] > signals['min_ADX'] * mult) & (signals['ADX'] > crit_level)) |
                ((signals['ADX'] > trend_level) & (signals['ADX'] < max_level))
            ) & (close < signals['avg_price']), 'sell', 'neutral'
        )
    )

    signals.drop(['ADX', 'min_ADX', 'avg_price'], axis=1, inplace=True)

    return signals
