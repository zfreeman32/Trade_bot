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

#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = trailing_stop_lx_signals(stock_df, trail_stop=5)  # Example: Use 5 periods for trailing stop

# Display output
print(signals_df.head())
