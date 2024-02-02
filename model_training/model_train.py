# %%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import ta 
from matplotlib import pyplot
from Strategies import call_Strategies
from model_training import preprocess_data
import keras_tuner as kt
from keras_tuner import HyperParameters
from model_training import model_build
seed = 42

#%%
# Load your OHLCV and indicator/strategies datasets
# Assuming df_ohlc is the OHLCV dataset and df_indicators is the indicators/strategies dataset
# Make sure your datasets are appropriately preprocessed before loading

spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = ta.add_all_ta_features(spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')

#%% 
# Assuming df is your DataFrame
all_signals_df = all_signals_df.loc[:, ~all_signals_df.columns.duplicated()]
# Keep the last occurrence of each column (drop_duplicates method)
all_signals_df = all_signals_df.T.drop_duplicates(keep='last').T

#%% 
for column in all_signals_df.columns:
	c = LabelEncoder().fit_transform(all_signals_df[column].astype(str))
	all_signals_df[column] = c

df = pd.concat([indicators_df, all_signals_df], axis = 1)
df = df.iloc[1000:7600,:]
columns_to_drop = ['trend_psar_up', 'trend_psar_down']
df = df.drop(columns=columns_to_drop)
df.head

reframed_data = preprocess_data.preprocess_stock_data(df)
reframed_data

#%%
from sklearn.model_selection import train_test_split

# Assuming values is your dataset
# split into train and test sets
values = reframed_data.values
# split into train and test sets without shuffling
train, test = train_test_split(values, test_size=0.2, shuffle=False)

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


#%%
# from keras import Sequential
# from keras.layers import LSTM, Dense

# # Define your LSTM model
# train_X = train_X[-5200:]
# train_y = train_y[-5200:]

# model = Sequential()
# model.add(LSTM(units=50, input_shape=(train_X.shape[1], train_X.shape[2])))
# model.add(Dense(units=1))  # Assuming regression task, adjust for classification
# model.compile(loss='mean_squared_error', optimizer='adam')

# # Train the model
# model.fit(train_X, train_y, epochs=5, batch_size=32, validation_data=(test_X, test_y))

# # Evaluate the model on the test set
# loss = model.evaluate(test_X, test_y)
# print(f'Test Loss: {loss}')

# # Make predictions on the test set
# predictions = model.predict(test_X)

#%%
# Hyperparameter tuning using RandomSearch from Kerastuner
tuner = kt.BayesianOptimization(
    hypermodel=model_build.build_LSTM_model,
    objective="val_loss",
    overwrite=True,
    max_retries_per_trial=3,
    max_consecutive_failed_trials=8,
)
train_X = train_X[-5200:]
train_y = train_y[-5200:]
print(train_X.shape)

# Train the model with hyperparameter tuning
tuner.search(train_X, train_y, epochs=5, validation_data=(test_X, test_y))

# Get the best model
best_model = tuner.get_best_models(1)[0]

# Evaluate the model
evaluation = best_model.evaluate(test_X, test_y)
print(f"Test Loss: {evaluation}")

#%%
# Get the best trial
best_trial = tuner.oracle.get_best_trials(1)[0]

# Get the best hyperparameters
best_hyperparameters = best_trial.hyperparameters.values
print("Best Hyperparameters:", best_hyperparameters)

# Print the summary of the best model
best_model.summary()

# fit network
history = best_model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
#%%

# make a prediction
yhat = best_model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

df
values = df.astype('float32')
values
# invert scaling for forecast
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# %%
