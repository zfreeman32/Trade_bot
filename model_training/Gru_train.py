#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
import ta 
from Strategies import call_Strategies
from model_training import preprocess_data
from numpy import concatenate
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

#%% Model 1 GRU

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

# design network
model1 = Sequential()
model1.add(GRU(50, input_shape=(train_X.shape[1], train_X.shape[2]), activation='tanh'))
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

#%% Model 2 GRU

[train_X, y_train, test_X, y_test] = preprocess_data.preprocess_stock_data(dataset=df, n_in = input_timesteps)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit the scaler on the training data
scaler.fit(train_X.reshape(-1, 1))

# design network
model2 = Sequential()
model2.add(GRU(50, dropout= 0.2, input_shape=(train_X.shape[1], train_X.shape[2]), activation='tanh'))
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

rmse1, mae1, inv_y1, inv_yhat1 = evaluate_model(model1, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 1', 'RMSE': rmse1, 'MAE': mae1})

rmse2, mae2, inv_y2, inv_yhat2 = evaluate_model(model2, test_X, y_test, scaler)
evaluation_results.append({'model': 'Model 2', 'RMSE': rmse2, 'MAE': mae2})

# Display results
for result in evaluation_results:
    print(f"{result['model']} - RMSE: {result['RMSE']:.3f}, MAE: {result['MAE']:.3f}")

# Plot predictions vs actual for one of the models (e.g., Model 1)
plt.plot(inv_y1, label='Actual')
plt.plot(inv_yhat1, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted for Model 1')
plt.show()