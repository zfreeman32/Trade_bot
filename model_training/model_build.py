# %%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Flatten, Input, Dense
from keras.optimizers import Adam
from layer_build import build_Dense_layer, build_LSTM_layer, build_GRU_layer, build_SimpleRNN_layer, build_Conv1D_layer, build_Dropout_layer, build_MaxPooling1D_Layer

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
    model.compile(optimizer=Adam(learning_rate=.001),
        loss='mean_squared_error')
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
					loss=mean_squared_error)
	return model

# Build the SimpleRNN model
def build_SimpleRNN_model(hp, input_shape, seed = 42):
	model = Sequential()
	# Add Input layer
	model.add(Input(shape=input_shape))  
	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_SimpleRNN_layers", min_value=1, max_value=3, step=1)):
		model.add(build_SimpleRNN_layer(hp, return_sequences=True, seed = seed))
	# Add last LSTM
	model.add(build_SimpleRNN_layer(hp, return_sequences= False, seed = seed))	
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp))
	model.add(Dense(units = 1))
	model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
					loss=mean_squared_error)
	return model

# Build Conv1D model
def build_Conv1D_model(hp, input_shape, data_format = 'channels_last'):
    model = Sequential()
    # Add Input layer
    model.add(Input(shape=input_shape))
    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Conv1D_layer(hp, data_format = data_format))  #"channels_last": (batch, steps, features), "channels_first": (batch, features, steps).    
    # Flatten layer before Dense layers
    model.add(Flatten())
    # Add Dense layers based on the hyperparameters
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
    model.add(Dense(units = 1))
    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),loss=mean_squared_error)
    return model

# Conv1D model
def build_Conv1DPooling_model(hp, input_shape, data_format = 'channel_last'):
    model = Sequential()
    # Add Input layer
    model.add(Input(shape=input_shape))
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
    return model

# Conv1D + LSTM model
def build_Conv1D_LSTM_model(hp, input_shape, data_format = 'channel_last'):
    model = Sequential()
    # Add Input layer
    model.add(Input(shape=input_shape))
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
    return model