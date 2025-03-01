# %%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from model_training.model_layers import build_Attention_layer, build_SeparableConv1D_layer, build_ConvLSTM2D_layer, build_MultiHeadAttention_layer, build_Dense_layer, build_LSTM_layer, build_GRU_layer, build_SimpleRNN_layer, build_Conv1D_layer, build_Dropout_layer, build_MaxPooling1D_Layer
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

seed = 42

#%%
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

def build_Transformer_model(hp, data_format='channels_last'):
    """
    Transformer-inspired architecture with MultiHeadAttention for time-series forecasting
    Captures long-range dependencies in sequence data
    
    Input shape: (batch_size, n_timesteps, n_features)
    Output shape: (batch_size, n_out)
    """
    model = Sequential()
    
    # Add positional encoding layer (implemented as a Lambda layer)
    model.add(tf.keras.layers.Lambda(
        lambda x: x + tf.cast(tf.math.sin(
            tf.range(tf.shape(x)[1], dtype=tf.float32)[None, :, None] * 
            (1000.0 ** (-tf.range(0, tf.shape(x)[2], dtype=tf.float32)[None, None, :] / tf.shape(x)[2]))
        ), dtype=tf.float32),
        input_shape=(None, None)  # Will be inferred from input
    ))
    
    # Add transformer blocks
    for i in range(hp.Int("num_transformer_blocks", min_value=1, max_value=4, step=1)):
        # Multi-head attention
        model.add(build_MultiHeadAttention_layer(hp))
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
        
        # Feed-forward network
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
    
    # Global average pooling to reduce sequence dimension
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    
    # Output layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    model.add(Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1)))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model

def build_BiLSTM_Attention_model(hp):
    """
    Bidirectional LSTM model with Attention mechanism
    Captures patterns in both forward and backward directions of the sequence
    
    Input shape: (batch_size, n_timesteps, n_features)
    Output shape: (batch_size, n_out)
    """
    model = Sequential()
    
    # Add bidirectional LSTM layers
    for i in range(hp.Int("num_bilstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = True if i < hp.Int("num_bilstm_layers", min_value=1, max_value=3, step=1) - 1 or i == 0 else False
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hp.Int(f"lstm_units_{i}", min_value=32, max_value=128, step=32),
                activation=hp.Choice(f"activation_{i}", ['relu', 'tanh', 'sigmoid']),
                return_sequences=return_sequences,
                dropout=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1),
                recurrent_dropout=hp.Float(f"rec_dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
            )
        ))
        
        # Add layer normalization after each BiLSTM layer
        model.add(tf.keras.layers.LayerNormalization())
    
    # Add attention layer
    model.add(build_Attention_layer(hp))
    
    # Add dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer
    model.add(Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1)))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model

def build_ConvLSTM2D_model(hp, data_format='channels_last'):
    """
    ConvLSTM2D model for capturing spatial-temporal patterns in time-series data
    Treats the time-series as a 2D spatial-temporal structure
    
    Input shape: Requires reshaping to (batch_size, time_steps, rows, cols, features)
    Output shape: (batch_size, n_out)
    """
    model = Sequential()
    
    # Add ConvLSTM2D layers
    for i in range(hp.Int("num_convlstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = True if i < hp.Int("num_convlstm_layers", min_value=1, max_value=3, step=1) - 1 else False
        model.add(build_ConvLSTM2D_layer(hp, return_sequences=return_sequences, data_format=data_format))
        model.add(build_Dropout_layer(hp))
    
    # Flatten the output
    model.add(Flatten())
    
    # Add dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer
    model.add(Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1)))
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    # Note: Using this model requires reshaping the input to 5D
    # e.g., (batch_size, timesteps, 1, n_features, 1) or another appropriate shape
    
    return model

def build_MultiStream_Hybrid_model(hp, data_format='channels_last'):
    """
    Multi-stream hybrid model that processes data through parallel streams
    and then combines them for final prediction
    
    Combines Conv1D, LSTM, and Attention mechanisms for capturing different aspects
    of the time-series data
    
    Input shape: (batch_size, n_timesteps, n_features)
    Output shape: (batch_size, n_out)
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=(None, None))  # Will be inferred from input
    
    # Stream 1: Convolutional stream
    conv_stream = input_layer
    for i in range(hp.Int("num_conv_layers", min_value=1, max_value=3, step=1)):
        conv_stream = build_Conv1D_layer(hp, data_format=data_format)(conv_stream)
        conv_stream = build_MaxPooling1D_Layer(hp, data_format=data_format)(conv_stream)
    
    # Stream 2: LSTM stream
    lstm_stream = input_layer
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = True if i < hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1) - 1 else False
        lstm_stream = build_LSTM_layer(hp, return_sequences=return_sequences)(lstm_stream)
    
    # Stream 3: Attention stream
    attention_stream = input_layer
    attention_stream = build_MultiHeadAttention_layer(hp)(
        attention_stream, attention_stream, attention_stream
    )
    
    # Flatten all streams
    conv_stream = Flatten()(conv_stream)
    if hp.Choice("flatten_lstm", values=[True, False]):
        lstm_stream = Flatten()(lstm_stream)
    attention_stream = Flatten()(attention_stream)
    
    # Concatenate the streams
    merged = tf.keras.layers.Concatenate()([conv_stream, lstm_stream, attention_stream])
    
    # Add dense layers after merge
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        merged = build_Dense_layer(hp)(merged)
        merged = build_Dropout_layer(hp)(merged)
    
    # Output layer
    output_layer = Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1))(merged)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model

def build_ResNet_model(hp, data_format='channels_last'):
    """
    ResNet-inspired architecture for time-series forecasting
    Uses skip connections to improve gradient flow and enable deeper networks
    
    Input shape: (batch_size, n_timesteps, n_features)
    Output shape: (batch_size, n_out)
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=(None, None))  # Will be inferred from input
    
    # Initial convolution
    x = build_Conv1D_layer(hp, data_format=data_format)(input_layer)
    
    # ResNet blocks
    for i in range(hp.Int("num_res_blocks", min_value=1, max_value=6, step=1)):
        # Store the input to the block for skip connection
        block_input = x
        
        # First conv layer in block
        x = build_SeparableConv1D_layer(hp, data_format=data_format)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Second conv layer in block
        x = build_SeparableConv1D_layer(hp, data_format=data_format)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Skip connection
        # If shapes don't match, use a 1x1 convolution to match dimensions
        if block_input.shape[-1] != x.shape[-1] or block_input.shape[-2] != x.shape[-2]:
            block_input = tf.keras.layers.Conv1D(
                filters=x.shape[-1], 
                kernel_size=1, 
                padding='same',
                data_format=data_format
            )(block_input)
        
        # Add skip connection
        x = tf.keras.layers.Add()([x, block_input])
        x = tf.keras.layers.Activation('relu')(x)
        
        # Optional pooling to reduce dimensions
        if hp.Choice(f"pool_after_block_{i}", values=[True, False]):
            x = build_MaxPooling1D_Layer(hp, data_format=data_format)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        x = build_Dense_layer(hp)(x)
        x = build_Dropout_layer(hp)(x)
    
    # Output layer
    output_layer = Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1))(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model

def build_TCN_model(hp, data_format='channels_last'):
    """
    Temporal Convolutional Network (TCN) model
    Uses dilated causal convolutions to capture long-range temporal dependencies
    
    Input shape: (batch_size, n_timesteps, n_features)
    Output shape: (batch_size, n_out)
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=(None, None))  # Will be inferred from input
    
    x = input_layer
    n_filters = hp.Int("n_filters", min_value=32, max_value=128, step=32)
    
    # TCN blocks with increasing dilation rates
    for i in range(hp.Int("num_tcn_blocks", min_value=1, max_value=6, step=1)):
        dilation_rate = 2**i  # Exponentially increasing dilation
        
        # Residual block
        # First dilated conv
        conv1 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu',
            data_format=data_format
        )(x)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = build_Dropout_layer(hp)(conv1)
        
        # Second dilated conv
        conv2 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu',
            data_format=data_format
        )(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = build_Dropout_layer(hp)(conv2)
        
        # Skip connection
        if x.shape[-1] != n_filters:
            x = tf.keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=1,
                padding='same',
                data_format=data_format
            )(x)
        
        x = tf.keras.layers.Add()([x, conv2])
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Fully connected layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        x = build_Dense_layer(hp)(x)
        x = build_Dropout_layer(hp)(x)
    
    # Output layer
    output_layer = Dense(units=hp.Int("output_units", min_value=1, max_value=15, step=1))(x)
    
    # Create the model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='mean_squared_error',
        metrics=['mse']
    )
    
    return model