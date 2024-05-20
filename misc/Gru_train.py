import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
import ta 
from Strategies import call_Strategies
from model_training import preprocess_data
from keras.layers import Dense
from keras_tuner.tuners import GridSearch 
from keras_tuner.engine.hyperparameters import HyperParameters
from keras.optimizers import Adam
import tensorflow as tf

seed = 42

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


# Load your OHLCV and indicator/strategies datasets
# Assuming df_ohlc is the OHLCV dataset and df_indicators is the indicators/strategies dataset
# Make sure your datasets are appropriately preprocessed before loading

spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
df = pd.concat([indicators_df, all_signals_df], axis = 1)

input_timesteps = 1
[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

hp = HyperParameters()

def build_model(hp):
    model = Sequential()
    model.add(GRU(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        return_sequences=True, 
        input_shape=(train_X.shape[1], train_X.shape[2])),
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )
    
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(GRU(
            units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
            return_sequences=True, 
            input_shape=(train_X.shape[1], train_X.shape[2])),
            activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
            recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
            use_bias=True,
            kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
            recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
            bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
            unit_forget_bias = hp.Boolean("forget_bias"),
            kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
            recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
            bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
            activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
            kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
            recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
            bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
            dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
            recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
            return_state=hp.Boolean("return_state"),
            go_backwards=hp.Boolean("go_backwards"),
            stateful=hp.Boolean("stateful"),
            unroll=hp.Boolean("unroll")
        )

    model.add(GRU(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        return_sequences=True, 
        input_shape=(train_X.shape[1], train_X.shape[2])),
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )
    
    if hp.Boolean("dropout"):
        model.add(Dropout(
            rate=hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1)))
    
    model.add(Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", [None, 'relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu']),
        use_bias=hp.Boolean("use_bias"),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm'])
    ))
    
    model.add(Dense(
        units=1, 
        activation=hp.Choice("output_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=hp.Boolean("use_bias"),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm'])
        ))
    
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["accuracy"],
    )
    
    return model


tuner = GridSearch(
    build_model,
    objective='val_loss',
    max_trials=5)

tuner.search(train_X, y_train, epochs=5, validation_data=(test_X, y_test))
best_model = tuner.get_best_models()[0]

print(best_model)
