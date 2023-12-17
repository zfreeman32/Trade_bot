#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ParameterGrid, GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, Attention
from sklearn.metrics import make_scorer
import ta 
from Strategies import call_Strategies
from model_training import preprocess_data
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

seed = 42

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

param_grid = {
    'units': [8, 16, 32, 64, 128],
    'activation': ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ],
    'seed': seed,
    'momentum': [0.9, 0.95],
    'epochs': [50, 100],
    'batch_size': [32, 64]
}

# Generate all possible combinations of hyperparameters
param_combinations = list(ParameterGrid(param_grid))

# Define a custom scoring function (you can use any metric you prefer)
def custom_scorer(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return -rmse

# design network
model1 = Sequential()
model1.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam')

# Create GridSearchCV
grid = GridSearchCV(estimator=model1, param_grid=param_grid, scoring=make_scorer(custom_scorer, greater_is_better=False),
                    cv=3, verbose=2, n_jobs=-1)
# Fit the grid search to the data
grid_result = grid.fit(train_X, y_train)
# Print the best parameters and the corresponding score
print("Best parameters found: ", grid_result.best_params_)
print("Best negative RMSE score: ", grid_result.best_score_)
# Get the best model
best_model = grid_result.best_estimator_.model

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

#%% Model 5 CNN - LSTM
# design network
model5 = Sequential()
model5.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
model5.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model5.add(Dense(1))
model5.compile(optimizer='adam', loss='mse')
# fit network
history = model5.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model5.predict(test_X)
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

#%% Model 6 LSTM - Attention
# design network
n_features = train_X.shape[2]
model6 = Sequential()
model6.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model6.add(Attention())
model6.add(Dense(1))
model6.compile(optimizer='adam', loss='mse')
# fit network
history = model6.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model6.predict(test_X)
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