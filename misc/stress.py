import pandas as pd
import numpy as np
from ta import volatility, trend

def stress_signals(stock_df, index_df, length=14, investment=10000, entry_level=30, 
                   exit_level=70, stop_loss=0.05, min_price=0.10, min_price_length=14, 
                   hedge_length=14, hedge_ratio=0.5):

    # Create a DataFrame for signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['close'] = stock_df['Close']

    # Calculate Stress Indicator
    signals['stress_indicator'] = calc_stress_indicator(stock_df['Close'], length)

    # Calculate the average close price of the index
    index_df['index_avg'] = trend.SMAIndicator(index_df['Close'], window=hedge_length).sma_indicator()

    # Identify the trend of index
    signals['index_trend'] = np.where(index_df['index_avg'] > index_df['index_avg'].shift(1), 'uptrend', 'downtrend')

    # Identify buy signals
    signals.loc[(signals['stress_indicator'] < entry_level) & (signals['index_trend'] == 'uptrend'), 'signal'] = 'buy'

    # Identify sell signals
    signals.loc[ (signals['stress_indicator'] > exit_level) | 
                 ((signals['close'] - signals['close'].shift(1)) / signals['close'].shift(1) < -stop_loss) |
                 (signals['close'].rolling(min_price_length).min() < signals['close'] * (1 - min_price)), 'signal'] = 'sell'

    return signals
#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = stress_signals(stock_df)  # Use 'Close' instead of 'sp500'

# Display output
print(signals_df.head())