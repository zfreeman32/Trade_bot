import pandas as pd
import numpy as np

def calculate_mid_range(price_high, price_low):
    return (price_high + price_low) / 2

def calculate_moving_average(data, length, average_type='simple'):
    if average_type == 'simple':
        return data.rolling(window=length).mean()
    elif average_type == 'exponential':
        return data.ewm(span=length).mean()
    elif average_type == 'weighted':
        weights = np.arange(1, len(data) + 1)
        return data.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    elif average_type == 'wilder':
        return data.ewm(alpha=1/length).mean()
    elif average_type == 'hull':
        return np.sqrt(data.rolling(window=int(np.sqrt(length))).mean()).ewm(span=int(np.sqrt(length))).mean()
    else:
        raise ValueError("Invalid average type. Expected one of: 'simple', 'exponential', 'weighted', 'wilder', 'hull'")

def MHLMA_signals(stock_df, length=50, high_low_length=15, average_type='simple'):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['MidRange'] = calculate_mid_range(stock_df['High'].rolling(window=high_low_length).max(), 
                                               stock_df['Low'].rolling(window=high_low_length).min())
    signals['MA'] = calculate_moving_average(stock_df['Close'], length, average_type)
    signals['MHL_MA'] = calculate_moving_average(stock_df['MidRange'], length, average_type)
    
    signals['MHLMA_signals'] = 0
    signals.loc[signals['MA'] > signals['MHL_MA'], 'MHLMA_signals'] = 1
    signals.loc[signals['MA'] < signals['MHL_MA'], 'MHLMA_signals'] = -1
    signals = signals.drop(['MA'], axis=1)

    return signals

