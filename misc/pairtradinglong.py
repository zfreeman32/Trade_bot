import pandas as pd
import numpy as np

# Pair Trading Long strategy
def pair_trading_long(primary_df, secondary_df, window_slow=25, window_fast=13, mode='trend following'):
    if len(primary_df) != len(secondary_df):
        raise ValueError("The primary and secondary dataframes must have the same length.")

    price_ratio = primary_df['Close'] / secondary_df['Close']
    sma_slow = price_ratio.rolling(window=window_slow).mean()
    sma_fast = price_ratio.rolling(window=window_fast).mean()

    signals = pd.DataFrame(index=primary_df.index)
    signals['price_ratio'] = price_ratio
    signals['sma_slow'] = sma_slow
    signals['sma_fast'] = sma_fast

    # Trend following mode
    if mode == 'trend following':
        signals['pair_trading_long_signals'] = 0
        signals.loc[(sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1)), 'pair_trading_long_signals'] = 1
        signals.loc[(sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1)), 'pair_trading_long_signals'] = -1

    # Mean reversion mode
    elif mode == 'mean reversion':
        signals['pair_trading_long_signals'] = 0
        signals.loc[(price_ratio < sma_fast) & (price_ratio.shift(1) >= sma_fast.shift(1)), 'pair_trading_long_signals'] = 1
        signals.loc[(price_ratio > sma_fast) & (price_ratio.shift(1) <= sma_fast.shift(1)), 'pair_trading_long_signals'] = -1

    # Trend-and-pullback mode
    elif mode == 'trend-and-pullback':
        signals['pair_trading_long_signals'] = 0
        signals.loc[(sma_fast < price_ratio) & (price_ratio < sma_slow), 'pair_trading_long_signals'] = 1
        signals.loc[(sma_fast > price_ratio) & (price_ratio > sma_slow), 'pair_trading_long_signals'] = -1

    signals.drop(['price_ratio', 'sma_slow', 'sma_fast'], axis=1, inplace=True)

    return signals
# Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = pair_trading_long(stock_df)

# Display output
print(signals_df.head())