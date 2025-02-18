import pandas as pd
import numpy as np
import pandas_ta as ta

# RSMK Strategy
def rsmk_signals(stock_df, corr_security_close, rs_length=10, average_length=10, time_exit_length=10):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the relative strength
    rs = ta.ratio(stock_df['Close'], corr_security_close).rolling(window=rs_length).mean()

    # Calculate exponential moving average of RS
    rs_ema = ta.ema(rs, length=average_length)

    # Create the RSMK signal
    signals['RSMK'] = rs_ema
    signals['rsmk_signal'] = 'neutral'
    signals.loc[(signals['RSMK'] > 0) & (signals['RSMK'].shift(1) <= 0), 'rsmk_signal'] = 'long'
    signals.loc[(signals.index - signals[signals['rsmk_signal'] == 'long'].index.shift(1)) >= time_exit_length, 'rsmk_signal'] = 'neutral'

    signals.drop(['RSMK'], axis=1, inplace=True)
    return signals

#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = rsmk_signals(stock_df)

# Display output
print(signals_df.head())
