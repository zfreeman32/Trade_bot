# %%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Flatten, Input, Dense
from keras.optimizers import Adam
from layer_build import build_MultiHeadAttention_layer, build_Dense_layer, build_LSTM_layer, build_GRU_layer, build_SimpleRNN_layer, build_Conv1D_layer, build_Dropout_layer, build_MaxPooling1D_Layer
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
seed = 42

# Build the LSTM model
def build_LSTM_model(hp):
    # input_shape = Input(shape=(train_X.shape[1], train_X.shape[2]))
    model = Sequential()
    # Add LSTM layers based on the hyperparameters
    for i in range(hp.Int("num_LSTM_layers", min_value=1, max_value=3, step=1)):
        model.add(build_LSTM_layer(hp, return_sequences=True))
    # Add last LSTM
    model.add(build_LSTM_layer(hp, return_sequences=False))
    # Add Dense layers based on the hyperparameters
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
    # Add output layer
    model.add(Dense(units=1))
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=.0001),
        loss='mean_squared_error', metrics=['accuracy'])
    return model

# Build the GRU model
def build_GRU_model(hp):
	model = Sequential()
	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_GRU_layers", min_value=1, max_value=3, step=1)):
		model.add(build_GRU_layer(hp, return_sequences=True))
	# Add last LSTM
	model.add(build_GRU_layer(hp, return_sequences=False))
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp))
	model.add(Dense(units = 1))
	model.compile(optimizer=Adam(learning_rate=.001),
		loss='mean_squared_error', metrics=['accuracy'])
	return model

# Build the SimpleRNN model
def build_SimpleRNN_model(hp):
	model = Sequential()
	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_SimpleRNN_layers", min_value=1, max_value=3, step=1)):
		model.add(build_SimpleRNN_layer(hp, return_sequences=True))
	# Add last LSTM
	model.add(build_SimpleRNN_layer(hp, return_sequences= False ))	
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp))
	model.add(Dense(units = 1))
	model.compile(optimizer=Adam(learning_rate=.001),
		loss='mean_squared_error', metrics=['accuracy'])
	return model

# Build Conv1D model
def build_Conv1D_model(hp, data_format='channels_last'):
    model = Sequential()
    num_conv1d_layers = hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)
    for i in range(num_conv1d_layers):
        model.add(build_Conv1D_layer(hp, data_format=data_format)) 
    model.add(Flatten())
    num_dense_layers = hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)
    for i in range(num_dense_layers):
        model.add(build_Dense_layer(hp))
    model.add(Dense(units=1))
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['accuracy'] 
    )
    
    return model

# Conv1D model
def build_Conv1DPooling_model(hp, data_format = 'channels_last'):
    model = Sequential()
    # Add Input layer
    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_layers", min_value=1, max_value=3, step=1)):
        for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
            model.add(build_Conv1D_layer(hp, data_format = data_format))
            # MaxPooling layer
        model.add(build_MaxPooling1D_Layer(hp, data_format = data_format)) # channels_last: (batch_size, steps, features), 'channels_first': (batch_size, features, steps)
    model.add(Flatten()),
    model.add(build_Dropout_layer(hp)),
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
    model.add(Dense(units = 1))
    model.compile(optimizer=Adam(learning_rate = .001))
    return model

# Conv1D + LSTM model
def build_Conv1D_LSTM_model(hp, data_format = 'channels_last'):
    model = Sequential()
    # Add Input layer
    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_layers_layers", min_value=1, max_value=3, step=1)):
        for i in range(hp.Int("num_conv1D_layers", min_value=1, max_value=3, step=1)):
            model.add(build_Conv1D_layer(hp, data_format = data_format))
        model.add(build_MaxPooling1D_Layer(hp, data_format = data_format)) # channels_last: (batch_size, steps, features), 'channels_first': (batch_size, features, steps)
    if hp.Boolean("Flatten"):
        model.add(Flatten()),
    # Add LSTM layers based on the hyperparameters
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        model.add(build_LSTM_layer(hp, return_sequences=True))
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
    model.add(Dense(units = 1))
    model.compile(optimizer=Adam(learning_rate = .001))
    return model

def build_LSTM_CNN_Hybrid_model(hp, data_format='channels_last'):
    """
    Hybrid model combining LSTM and CNN for capturing both temporal dependencies and local patterns
    """
    model = Sequential()
    model.add(tf.keras.layers.LSTM(
        units=hp.Int("lstm_units", min_value=64, max_value=256, step=32),
        return_sequences=True,
        input_shape=(1, 128)
    ))
    model.add(build_Dropout_layer(hp))
    model.add(tf.keras.layers.Reshape((1, 64)))
    for i in range(hp.Int("num_conv_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Conv1D_layer(hp))
        model.add(build_MaxPooling1D_Layer(hp))
    model.add(Flatten())
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    model.add(Dense(2))
    def custom_loss(y_true, y_pred):
        prediction, confidence = tf.split(y_pred, 2, axis=1)
        mse = tf.keras.losses.mean_squared_error(y_true, prediction)
        confidence_penalty = 0.1 * tf.reduce_mean(tf.abs(confidence))
        return mse + confidence_penalty
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=custom_loss,
        metrics=['mse']
    )
    return model

def build_Attention_LSTM_model(hp):
    """
    LSTM model with attention mechanism for focusing on relevant time steps
    """
    model = Sequential()
    model.add(tf.keras.layers.LSTM(
        units=hp.Int("lstm_units", min_value=64, max_value=256, step=32),
        return_sequences=True,
        input_shape=(1, 249)
    ))
    model.add(build_MultiHeadAttention_layer(hp))
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        model.add(tf.keras.layers.LSTM(
            units=hp.Int(f"lstm_units_{i}", min_value=32, max_value=128, step=32),
            return_sequences=i < hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1) - 1
        ))
        model.add(build_Dropout_layer(hp))
    model.add(Flatten())
    prev_layer = model.layers[-1].output
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        dense = build_Dense_layer(hp)(prev_layer)
        dropout = build_Dropout_layer(hp)(dense)
        if i > 0:  # Add residual connection after first layer
            prev_layer = tf.keras.layers.Add()([prev_layer, dropout])
        else:
            prev_layer = dropout
    model.add(Dense(2))  # Prediction and confidence interval
    def attention_guided_loss(y_true, y_pred):
        prediction, confidence = tf.split(y_pred, 2, axis=1)
        weighted_mse = tf.reduce_mean(tf.square(y_true - prediction) * tf.exp(-confidence))
        confidence_regularization = 0.1 * tf.reduce_mean(confidence)
        return weighted_mse + confidence_regularization
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=attention_guided_loss,
        metrics=['mse']
    )
    return model

def build_RandomForestRegressor_model(hp):
    """
    Builds an optimal RandomForestRegressor model
    """
    model = RandomForestRegressor(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        min_samples_split=hp.Int("min_samples_split", min_value=2, max_value=10, step=2),
        min_samples_leaf=hp.Int("min_samples_leaf", min_value=1, max_value=5, step=1),
        max_features=hp.Choice("max_features", ["auto", "sqrt", "log2"]),
        random_state=42,
        n_jobs=-1
    )
    return model

def build_XGBoostRegressor_model(hp):
    """
    Builds an optimal XGBoost Regressor model
    Does not inherently handle time-series (feature engineering required)
    """
    model = XGBRegressor(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        subsample=hp.Float("subsample", min_value=0.5, max_value=1.0, step=0.1),
        colsample_bytree=hp.Float("colsample_bytree", min_value=0.5, max_value=1.0, step=0.1),
        random_state=42,
        n_jobs=-1
    )
    return model

def build_GradientBoostingRegressor_model(hp):
    """
    Builds an optimal Gradient Boosting Regressor model
    """
    model = GradientBoostingRegressor(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        min_samples_split=hp.Int("min_samples_split", min_value=2, max_value=10, step=2),
        min_samples_leaf=hp.Int("min_samples_leaf", min_value=1, max_value=5, step=1),
        random_state=42
    )
    return model

def build_LightGBMRegressor_model(hp):
    """
    Builds an optimal LightGBM Regressor model
    """
    model = LGBMRegressor(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        num_leaves=hp.Int("num_leaves", min_value=20, max_value=150, step=10),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        subsample=hp.Float("subsample", min_value=0.5, max_value=1.0, step=0.1),
        colsample_bytree=hp.Float("colsample_bytree", min_value=0.5, max_value=1.0, step=0.1),
        random_state=42,
        n_jobs=-1
    )
    return model

def build_CatBoostRegressor_model(hp):
    """
    Builds an optimal CatBoost Regressor model
    """
    model = CatBoostRegressor(
        iterations=hp.Int("iterations", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        depth=hp.Int("depth", min_value=3, max_value=15, step=3),
        l2_leaf_reg=hp.Float("l2_leaf_reg", min_value=1, max_value=10, step=1),
        random_state=42,
        verbose=0
    )
    return model
