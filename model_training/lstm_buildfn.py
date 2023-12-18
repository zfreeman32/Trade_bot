import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import ta 
from Strategies import call_Strategies
from model_training import preprocess_data
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras_tuner.tuners import GridSearch 
from keras_tuner.engine.hyperparameters import HyperParameters
from keras.optimizers import Adam
import tensorflow as tf

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

hp = HyperParameters()

def build_model(hp):
    model = Sequential()
    model.add(LSTM(
        hp.Int("units",min_value=32,max_value=512,step=32), 
        return_sequences=True, 
        input_shape=(train_X.shape[1],train_X.shape[2])))
    for i in range(hp.Int("n_layers", 1, 3)):
        model.add(LSTM(
            units=hp.Int("units", min_value=32, max_value=512, step=32), 
            activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
            input_shape=(train_X.shape[1], train_X.shape[2]),
            return_sequences= True
            )
        )
        model.add(Dense(
            hp.Int("units", min_value=8, max_value=32, step=8),
            activation='relu'))
        
    model.add(LSTM(
            units=hp.Int("units", min_value=32, max_value=512, step=32), 
            activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
            input_shape=(train_X.shape[1], train_X.shape[2])
         )
    )    
    if hp.Boolean("dropout"):
        model.add(Dropout(
            rate=hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(
        units = 1, 
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ])))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["accuracy"],
    )
    return model

tuner = GridSearch (
    build_model,
    objective='val_accuracy',
    max_trials=5)

tuner.search(train_X, y_train, epochs=5, validation_data=(test_X, y_test))
best_model = tuner.get_best_models()[0]

print(best_model)
