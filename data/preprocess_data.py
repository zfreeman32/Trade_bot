import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from features import call_Strategies
from data import preprocess_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame, concat
import features.all_indicators as all_indicators

# Function to frame the dataset as supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Combine all
    agg = concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# process data
# Generate indicators
def generate_features(data):
    df_with_indicators = all_indicators.generate_all_indicators(data)

    # Generate Signals
    df_strategies = pd.DataFrame(data).reset_index(drop=True)
    print(df_strategies.head())

    # Now pass the DataFrame to the generate_all_signals function
    df_strategies = call_Strategies.generate_all_signals(df_strategies)
    df_strategies = df_strategies.drop(['kama_signal'],axis=1)

    # Convert columns to string
    data['Date'] = data['Date'].astype(str)  
    data['Time'] = data['Time'].astype(str)
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

    # Identify common columns except 'Date'
    common_cols = [col for col in df_strategies.columns if col in df_with_indicators.columns and col != 'Date']

    # Remove common columns from df2 before merging
    df_with_indicators = df_with_indicators.drop(columns=common_cols)

    # Then do your merge
    df_strategies = df_strategies.set_index('Date')
    df_with_indicators = df_with_indicators.set_index('Date')
    merged_df = pd.concat([df_strategies, df_with_indicators], axis=1).reset_index()
    df = pd.DataFrame(merged_df)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')  # Convert to datetime
    df['Minutes'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 60  
    df.drop(columns=['Time'], inplace=True)

    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Remove 'Time' and 'Date' from non_numeric_cols if they exist
    cols_to_encode = [col for col in non_numeric_cols if col not in ['Time']]

    # Initialize LabelEncoder
    label_encoders = {}
    for col in cols_to_encode:
        # Replace NaN values with 0
        df[col].fillna(0, inplace=True)
        df[col].replace("", 0, inplace=True)
        df[col] = df[col].fillna('').apply(str).str.strip() 
        print(df[col].shape) # Convert all values to string and strip whitespace
        le = LabelEncoder()
        df[col]=df[col].astype(str)
        df[col] = le.fit_transform(df[col]) 
        label_encoders[col] = le 

    df.fillna(-9999, inplace=True)
    return df

#%% usage

# file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\sampled_EURUSD_1min.csv'
# data = pd.read_csv(
#     file_path, 
#     header=0
# )
# data = data.tail(1000)

# df = preprocess_data(data)
# features = df.drop(columns=['Close'])  
# target = df[['Close']] 

# # Frame the dataset for multi-step forecasting
# n_in = 240  # Number of past observations (1 Month)
# n_out = 15  # Number of future observations to forecast (1 Day)
# supervised_features = series_to_supervised(features.values, n_in=n_in, n_out=n_out)
# supervised_target = series_to_supervised(target.values, n_in=n_in, n_out=n_out)

# # Extract input (X) and output (y)
# n_features = features.shape[1]  # Number of features excluding 'Close'
# n_obs = n_in * n_features  # Total past observations used as input

# X = supervised_features.iloc[:, :n_obs].values  # Features (past observations)
# y = supervised_target.iloc[:, n_in * target.shape[1]:].values  # Target ('Close' values)

# print("Features (X) shape:", X.shape)
# print("Target (y) shape:", y.shape)

# #%%
# import dask.array as da
# import dask.dataframe as dd
# import numpy as np
# from dask_ml.model_selection import train_test_split
# import zarr
# from numpy.lib.stride_tricks import sliding_window_view

# def process_large_dataset(file_path, window_size=120, chunk_size=1000):
#     """
#     Process large datasets efficiently using chunked processing and memory optimization.
    
#     Parameters:
#     -----------
#     file_path : str
#         Path to the CSV file containing the trading data
#     window_size : int
#         Size of the sliding window for feature generation
#     chunk_size : int
#         Size of chunks for processing
        
#     Returns:
#     --------
#     train_X, test_X, train_y, test_y : dask arrays
#         Training and testing splits of features and targets
#     """
#     # Read data in chunks using dask
#     ddf = dd.read_csv(file_path)
    
#     # Process features in chunks
#     def process_chunk(df_chunk):
#         # Generate features for the chunk
#         chunk_features = preprocess_data.generate_features(df_chunk)
#         chunk_features[['buy_signal']] = chunk_features[['buy_signal']].astype(np.float32).shift(-1).fillna(0)
#         return chunk_features.astype(np.float32)
    
#     # Apply processing to each chunk
#     df = ddf.map_partitions(process_chunk).compute()
    
#     # Create zarr array for out-of-memory storage
#     store = zarr.TempStore()
#     feature_store = zarr.zeros((len(df) - window_size + 1, window_size, df.shape[1] - 3),
#                               chunks=(chunk_size, window_size, -1),
#                               dtype=np.float32,
#                               store=store)
    
#     # Process data in chunks
#     for i in range(0, len(df) - window_size + 1, chunk_size):
#         end_idx = min(i + chunk_size, len(df) - window_size + 1)
#         chunk_data = df.iloc[i:i + window_size + chunk_size - 1]
        
#         # Generate sliding windows for the chunk
#         chunk_features = sliding_window_view(
#             chunk_data.drop(columns=['buy_signal', 'sell_signal', 'Close_Position']).values,
#             window_size,
#             axis=0
#         )
        
#         # Store in zarr array
#         feature_store[i:end_idx] = chunk_features[:end_idx-i]
    
#     # Create dask array from zarr storage
#     features_array = da.from_zarr(feature_store)
#     target_array = da.from_array(df['buy_signal'][window_size:].values, 
#                                 chunks=chunk_size)
    
#     # Ensure arrays have matching lengths
#     features_array = features_array[:target_array.shape[0]]
    
#     # Perform train-test split
#     train_X, test_X, train_y, test_y = train_test_split(
#         features_array, target_array,
#         test_size=0.2,
#         random_state=42,
#         shuffle=False
#     )
    
#     return train_X, test_X, train_y, test_y

# file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv'
# train_X, test_X, train_y, test_y = process_large_dataset(file_path)