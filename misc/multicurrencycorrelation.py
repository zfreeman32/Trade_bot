import pandas as pd
import numpy as np
from scipy.stats import pearsonr

def multi_currency_correlation(df1, df2, multiplier, lower_limit, upper_limit):
    # Calculate the correlation between the two currency pairs
    correlation, _ = pearsonr(df1['Close'], df2['Close'])

    # Initialize an empty DataFrame for the signals
    signals = pd.DataFrame(index=df1.index)
    signals['Correlation'] = correlation * multiplier
    signals['signal'] = 0.0

    # Generate the buy and sell signals based on the correlation and signal limits
    signals['signal'][signals['Correlation'] > upper_limit] = -1.0
    signals['signal'][signals['Correlation'] < lower_limit] = 1.0

    return signals
# Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = multi_currency_correlation(stock_df)

# Display output
print(signals_df.head())