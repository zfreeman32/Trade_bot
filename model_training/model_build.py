# %%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from math import sqrt
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
import ta 
from matplotlib import pyplot
from pandas import DataFrame, concat
from Strategies import call_Strategies
from model_training import preprocess_data
# from keras.models import Sequential
# from keras_tuner.tuners import GridSearch 
# from keras.optimizers import Adam
# import tensorflow as tf
# from layer_build import build_Dense_layer, build_LSTM_layer

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
for column in all_signals_df.columns:
	k = all_signals_df[column].astype(str)
	d = k.iloc[:,1].values
	c = LabelEncoder().fit_transform(d)
	all_signals_df[column] = c

df = pd.concat([indicators_df, all_signals_df], axis = 1)
df = df.iloc[1000:7600,:]
df.head

#%%
reframed_data = preprocess_data.preprocess_stock_data(df)
reframed_data

#%%
# split into train and test sets
values = reframed_data.values
n_total_days = 1000
n_train_days = int(n_total_days * 0.8)
# Use the last 1000 days
data_subset = values[-n_total_days:, :]
# split into training and testing
train = data_subset[:n_train_days, :]
test = data_subset[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#%%
# Build the LSTM model
def build_model(hp):
	model = Sequential()

	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
		model.add(build_LSTM_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

	# Add last LSTM
	model.add(build_LSTM_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
		
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
					loss=mean_squared_error)
	return model

# Hyperparameter tuning using RandomSearch from Kerastuner
tuner = GridSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Adjust as needed
    directory='hyperparameter_tuning',
    project_name='lstm_hyperparameter_tuning'
)

# Train the model with hyperparameter tuning
tuner.search(train_X, train_y, epochs=10, validation_data=(test_X, test_y))

# Get the best model
best_model = tuner.get_best_models(1)[0]

# Evaluate the model
evaluation = best_model.evaluate(test_X, test_y)
print(f"Test Loss: {evaluation}")

# fit network
history = best_model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = best_model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
scaler = MinMaxScaler(feature_range=(0, 1))
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