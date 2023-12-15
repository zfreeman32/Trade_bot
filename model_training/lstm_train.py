#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import ta 
from Strategies import call_Strategies
from model_training import preprocess_data
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

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

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

param_grid = {
    'nodes_hidden_layers': [(32,), (64,), (32, 64)],
    'units': [8, 16, 32, 64, 128],
    'dropout_layers': [0.2, 0.5],
    'weight_initializer': ['glorot_uniform', 'orthogonal'],
    'activation_ function': ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ],
    'momentum': [0.9, 0.95],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

# Generate all possible combinations of hyperparameters
param_combinations = list(ParameterGrid(param_grid))

# design network
model1 = Sequential()
model1.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam')
# fit network
history = model1.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model1.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = y_test.reshape((len(y_test), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#%% Model 2 with varying dense layrs
# design network
model2 = Sequential()
model2.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(Dense(1))
model2.add(Dense(1))
model2.compile(loss='mae', optimizer='adam')
# fit network
history = model2.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model2.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = y_test.reshape((len(y_test), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#%% Model 3 with dropout
# design network
model3 = Sequential()
model3.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model3.add(Dense(1))
model3.add(Dropout(0.5))
model3.compile(loss='mae', optimizer='adam')
# fit network
history = model3.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model3.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = y_test.reshape((len(y_test), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#%% Model 4 multiple LSTM
# design network
model4 = Sequential()
model4.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model4.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model4.add(Dense(1))
model4.compile(loss='mae', optimizer='adam')
# fit network
history = model4.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model4.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = y_test.reshape((len(y_test), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

#%%
# Define a matrix of hyperparameters to search


# best_model = None
# best_rmse = float('inf')

# for params in param_combinations:
#     model = Sequential()

#     # Add LSTM layers
#     for nodes in params['nodes_hidden_layers']:
#         model.add(LSTM(nodes, activation=params['activation_function'], input_shape=(train_X.shape[1], train_X.shape[2])))

#     # Add Dense layer
#     model.add(Dense(params['units_dense_layer'], activation=params['activation_function']))

#     # Add Dropout layers
#     for dropout in params['dropout_layers']:
#         model.add(Dropout(dropout))

#     # Compile the model
#     sgd = SGD(learning_rate=0.01, momentum=params['momentum'])
#     model.compile(optimizer=sgd, loss='mean_squared_error')

#     # Early stopping to prevent overfitting
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     # Train the model
#     history = model.fit(train_X, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
#                         validation_data=(test_X, y_test), callbacks=[early_stopping], verbose=0)

#     # Make predictions on the test set
#     y_pred = model.predict(test_X)

#     # Calculate RMSE
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#     # Print the RMSE for the current set of hyperparameters
#     print(f"Hyperparameters: {params}, RMSE: {rmse}")

#     # Update the best model if the current model is better
#     if rmse < best_rmse:
#         best_rmse = rmse
#         best_model = model

# # Print the hyperparameters of the best model
# print(f"Best Hyperparameters: {best_model.get_config()}")







# Design network
# def build_lstm_model(units=50, activation='tanh', recurrent_activation='sigmoid',
#                      use_bias=True, kernel_initializer='glorot_uniform',
#                      recurrent_initializer='orthogonal', bias_initializer='zeros',
#                      unit_forget_bias=True, kernel_regularizer=None,
#                      recurrent_regularizer=None, bias_regularizer=None,
#                      activity_regularizer=None, kernel_constraint=None,
#                      recurrent_constraint=None, bias_constraint=None,
#                      dropout=0, recurrent_dropout=0, seed=None,
#                      return_sequences=False, return_state=False,
#                      go_backwards=False, stateful=False, unroll=False):
#     model = Sequential()
#     model.add(LSTM(units=units,
#                    activation=activation,
#                    recurrent_activation=recurrent_activation,
#                    use_bias=use_bias,
#                    kernel_initializer=kernel_initializer,
#                    recurrent_initializer=recurrent_initializer,
#                    bias_initializer=bias_initializer,
#                    unit_forget_bias=unit_forget_bias,
#                    kernel_regularizer=kernel_regularizer,
#                    recurrent_regularizer=recurrent_regularizer,
#                    bias_regularizer=bias_regularizer,
#                    activity_regularizer=activity_regularizer,
#                    kernel_constraint=kernel_constraint,
#                    recurrent_constraint=recurrent_constraint,
#                    bias_constraint=bias_constraint,
#                    dropout=dropout,
#                    recurrent_dropout=recurrent_dropout,
#                    seed=seed,
#                    return_sequences=return_sequences,
#                    return_state=return_state,
#                    go_backwards=go_backwards,
#                    stateful=stateful,
#                    unroll=unroll))
    
#     # Add a Dense layer for output
#     model.add(Dense(1))  # Assuming you're predicting a single value (regression)
    
#     model.compile(optimizer='adam', loss='mean_squared_error')  # You can change the loss function based on your problem type
    
#     return model

# fit network
# history = model.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# # plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
 
# # make a prediction
# yhat = model.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:,0]
# # invert scaling for actual
# test_y = y_test.reshape((len(y_test), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:,0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)