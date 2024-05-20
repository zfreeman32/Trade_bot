#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler

'''
Input a data frame containing OHLCV stock data and all features, 
encode any categorcial columns. 
n_in is the number of days back to use as input. 
n_out is the number of predictions to make. 
format is the drmat f your 'Date' column. Default: format='%m/%d/%Y'
Returns an aggregated dataframe with the Close (target) variable as the 
last column, all columns scaled.
'''
#%%
# filepath is Date, Open, High, Llow, Close, Volume dataset
def preprocess_stock_data(df, n_in=1, n_out=1, format='%m/%d/%Y'):

    # convert series to supervised learning
    def series_to_supervised(data, n_in=n_in, n_out=n_out):
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
        return agg

    #%%
    # Extract year, month, and day from the 'Date' column
    date_column = df['Date']
    date_df = pd.to_datetime(date_column, format=format)
    # Create new columns for year, month, and day
    df['Year'] = date_df.dt.year
    df['Month'] = date_df.dt.month
    df['Day'] = date_df.dt.day
    # Drop the original 'Date' column
    df.drop(columns=['Date'], inplace=True)

    #%%
    # ensure all data is float
    values = df.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    #%%
    half_reframed = len(reframed.columns) * n_in
    reframed_data = reframed.iloc[:, :half_reframed].copy()
    # Save the 'var4(t)' (CLose) (Target) column
    var4_column = reframed['var4(t)'].copy()
    # Attach the 'var4(t)' column to the end of the reframed data
    reframed_data['var4(t)'] = var4_column
    # Print the modified reframed data
    print(reframed_data.head())

    return reframed_data
