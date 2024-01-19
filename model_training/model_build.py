# %%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, Dense
from keras.layers import Flatten, MaxPooling1D
from keras.optimizers import Adam
from layer_build import build_Dense_layer, build_LSTM_layer, build_GRU_layer, build_SimpleRNN_layer, build_Conv1D_layer, build_Dropout_layer

seed = 42

#%%
# Build the LSTM model
def build_LSTM_model(hp):
	model = Sequential()

	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
		model.add(build_LSTM_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

	# Add last LSTM
	model.add(build_LSTM_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
		
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
					loss=mean_squared_error)
	return model

#%%
# Build the GRU model
def build_GRU_model(hp):
	model = Sequential()
	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
		model.add(build_GRU_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
	# Add last LSTM
	model.add(build_GRU_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
	model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
					loss=mean_squared_error)
	return model

#%%
# Build the SimpleRNN model
def build_SimpleRNN_model(hp):
	model = Sequential()

	# Add LSTM layers based on the hyperparameters
	for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
		model.add(build_SimpleRNN_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

	# Add last LSTM
	model.add(build_SimpleRNN_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
		
	# Add Dense layers based on the hyperparameters
	for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
		model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))

	model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
					loss=mean_squared_error)
	
	return model

#%%
# Build Conv1D model
def build_Conv1D_model(hp):
    model = Sequential()

    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Conv1D_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
		
    # Flatten layer before Dense layers
    model.add(Flatten())

    # Add Dense layers based on the hyperparameters
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.compile(optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="LOG")),
                  loss=mean_squared_error)

    return model

# Conv1D model
def build_Conv1DPooling_model(hp, input_shape):
    model = Sequential()
    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
        for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
            model.add(build_Conv1D_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format=None))
    model.add(Flatten()),
    model.add(build_Dropout_layer(hp)),

    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))
    
    model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))
    return model

# Conv1D + LSTM model
def build_Conv1D_LSTM_model(hp):
    model = Sequential()
    # Add Conv1D layers based on the hyperparameters
    for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
        for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
            model.add(build_Conv1D_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid", data_format=None))

    if hp.Boolean("Flatten"):
        model.add(Flatten()),
    # Add LSTM layers based on the hyperparameters
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        model.add(build_LSTM_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))

    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(Dense(units = 1, input_shape=(train_X.shape[1], train_X.shape[2])))
    return model