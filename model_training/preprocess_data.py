#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import numpy as np
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


#%%
# filepath is Date, Open, High, Llow, Close, Volume dataset
def preprocess_stock_data(dataset, n_in=1, n_out=1, datecolumn = 3, dropnan=True):

    # convert series to supervised learning
    def series_to_supervised(data, n_in=n_in, n_out=n_out, dropnan=dropnan):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # load dataset
    dataset = read_csv(filepath)
    values = dataset.values
    # Extract year, month, and day from the date column
    date_column = values[:, 0]
    date_df = pd.to_datetime(date_column, format='%m/%d/%Y')
    datevalues = np.column_stack((values[:, :0], date_df.month, date_df.day, date_df.year))
    # Concatenate datevalues to the front of values
    values = np.concatenate((datevalues, values), axis=1)
    # Drop the 4th column
    values = np.delete(values, datecolumn, axis=1)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, n_out, dropnan)
    reframed_close = datecolumn + 4
    # Keep only the 'Close' column (column index 7) for prediction
    reframed_target = reframed.iloc[:, [reframed_close]]
    # split into train and test sets
    y = reframed_target.values
    x = reframed.drop(columns=reframed_target).values
    # Train-test split
    # Create an index array based on the length of your data
    data_length = len(x)
    index_array = np.arange(data_length)
    # Sort the index array
    sorted_index_array = np.argsort(index_array)
    # Use the sorted index array to create the train-test split
    train_indices, test_indices = train_test_split(sorted_index_array, test_size=0.2, random_state=42)
    # Use the indices to create the actual train-test split
    X_train, X_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    test_X = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(train_X.shape, y_train.shape, test_X.shape, y_test.shape)

    return train_X, y_train, test_X, y_test

#%%
filepath = r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv'
[train_X, y_train, test_X, y_test] = preprocess_stock_data(dataset=filepath)
