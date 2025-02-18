import pandas as pd
from ta import trend

def simple_trend_channel(stock_df, trend_check='normal', earnings_check=2):
    signals = pd.DataFrame(index=stock_df.index)

    # Calculate moving averages
    average_50 = stock_df['Close'].rolling(window=50).mean()
    fast_average = stock_df['Close'].rolling(window=20).mean()
    slow_average = stock_df['Close'].rolling(window=200).mean()
    upper_band = stock_df['High'].rolling(window=8).mean()
    lower_band = stock_df['Low'].rolling(window=8).mean()

    # Create default signals
    signals['buy_to_open'] = 0
    signals['sell_to_close'] = 0
    signals['sell_to_open'] = 0
    signals['buy_to_close'] = 0
    
    # Iterate over the dataframe
    for i in range(1, len(stock_df)):
        # check for earnings announcement
        if stock_df['Earnings Date'].iloc[i] > earnings_check:
            continue
            
        # Buy to open condition
        if (stock_df['Close'].iloc[i] > upper_band.iloc[i]) and (fast_average.iloc[i] > average_50.iloc[i]):
            signals['buy_to_open'].iloc[i] = 1
        
        # Sell to close condition
        if (stock_df['Close'].iloc[i] < lower_band.iloc[i]) or (stock_df['Earnings Date'].iloc[i] == 1):
            signals['sell_to_close'].iloc[i] = 1
            
        # Sell to open condition
        if (stock_df['Close'].iloc[i] < lower_band.iloc[i]) and (fast_average.iloc[i] < average_50.iloc[i]):
            signals['sell_to_open'].iloc[i] = 1
            
        # Buy to close condition
        if (stock_df['Close'].iloc[i] > upper_band.iloc[i]) or (stock_df['Earnings Date'].iloc[i] == 1):
            signals['buy_to_close'].iloc[i] = 1
            
    return signals
#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = simple_trend_channel(stock_df)

# Display output
print(signals_df.head())
