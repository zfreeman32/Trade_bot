#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import Input
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

# design network
model1 = Sequential()
model1.add(Input(shape=(train_X.shape[1], train_X.shape[2])))
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


# %% Model 2 with varying dense layrs

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

# design network
model2 = Sequential()
model2.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(Dense(25))
model2.add(Dense(1))
model2.compile(loss='mae', optimizer='adam')
# fit network
history = model2.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# make a prediction
yhat = model2.predict(test_X)

# Reshape yhat to be 2D (if it's not already)
yhat = yhat.reshape((yhat.shape[0], yhat.shape[1]))

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 0, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = y_test.reshape((len(y_test), 1))
inv_y = concatenate((test_y, test_X[:, 0, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]


print(train_X.shape)

#%% Model 3 with dropout

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

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

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

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

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

# design network
# model5 = Sequential()
# model5.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_X.shape[1], train_X.shape[2])))
# model5.add(Flatten())  # Flatten layer added
# model5.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model5.add(Dense(1))
# model5.compile(optimizer='adam', loss='mse')
# # fit network
# history = model5.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# # make a prediction
# yhat = model5.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# test_y = y_test.reshape((len(y_test), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

#%% Model 6 LSTM - Attention

# [train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# # normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# # Fit the scaler on the training data
# scaler.fit(train_X.reshape(-1, 1))

# # design network
# n_features = train_X.shape[2]
# model6 = Sequential()
# model6.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
# model6.add(Attention())
# model6.add(Dense(1))
# model6.compile(optimizer='adam', loss='mse')
# # fit network
# history = model6.fit(train_X, y_train, epochs=50, batch_size=72, validation_data=(test_X, y_test), verbose=2, shuffle=False)
# # make a prediction
# yhat = model6.predict(test_X)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# # invert scaling for actual
# test_y = y_test.reshape((len(y_test), 1))
# inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]

#%%
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model, test_X, y_test, scaler):
    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    # invert scaling for actual
    test_y = y_test.reshape((len(y_test), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    mae = mean_absolute_error(inv_y, inv_yhat)

    return rmse, mae, inv_y, inv_yhat

# List to store the evaluation results of each model
evaluation_results = []

# Evaluate Model 1
rmse1, mae1, inv_y1, inv_yhat1 = evaluate_model(model1, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 1', 'RMSE': rmse1, 'MAE': mae1})

# Evaluate Model 2
rmse2, mae2, inv_y2, inv_yhat2 = evaluate_model(model2, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 2', 'RMSE': rmse2, 'MAE': mae2})

# Evaluate Model 3
rmse3, mae3, inv_y3, inv_yhat3 = evaluate_model(model3, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 3', 'RMSE': rmse3, 'MAE': mae3}
                          )
rmse4, mae4, inv_y4, inv_yhat4 = evaluate_model(model4, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 4', 'RMSE': rmse4, 'MAE': mae4})

# Display results
for result in evaluation_results:
    print(f"{result['model']} - RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}")

# Plot predictions vs actual for one of the models (e.g., Model 1)
plt.plot(inv_y1, label='Actual')
plt.plot(inv_yhat1, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted for Model 1')
plt.show()
