import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dropout, Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import ta
from Strategies import call_Strategies
import pandas as pd
from sklearn.model_selection import train_test_split

spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
df = pd.concat([indicators_df, signals_df], axis = 1)

def preprocess_data(df, target_col='Close', sequence_length=10, test_size=0.2):
    """
    Preprocess the concatenated dataframe for training and testing.

    Parameters:
    - df: Concatenated dataframe with OHLCV, indicators, and signals.
    - target_col: Name of the target variable (e.g., 'Close').
    - sequence_length: Length of sequences for time series data.
    - test_size: Fraction of the data to be used for testing.

    Returns:
    - X_train, X_test, y_train, y_test: Training and testing sets for features and target.
    """

    # Extract features (excluding 'Close') and target variable
    features = df[df.columns.difference([target_col])].values
    target = df[target_col].values.reshape(-1, 1)

    # Combine features and target
    data = pd.concat([df[['Open', 'High', 'Low', 'Close', 'Volume']], pd.DataFrame(features), pd.DataFrame(target)], axis=1)

    # Create sequences for time series data
    sequences = [data.iloc[i:i + sequence_length] for i in range(len(data) - sequence_length + 1)]
    sequences = pd.concat(sequences, ignore_index=True)

    # Split the data into train and test sets
    train_size = int(len(sequences) * (1 - test_size))
    train, test = sequences[:train_size], sequences[train_size:]

    # Split into features and target
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(df, sequence_length=10, test_size=0.2)
print(X_train)
print(X_train.shape)

def build_lstm_model(neurons=50, dropout_rate=0.2):
    X_train_reshaped = X_train.values.reshape(32, X_train.shape[0], X_train.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_simplernn_model(neurons=50, dropout_rate=0.2):
    model = Sequential()
    model.add(SimpleRNN(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def build_gru_model(neurons=50, dropout_rate=0.2):
    model = Sequential()
    model.add(GRU(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Wrap the Keras models in scikit-learn compatible wrappers
lstm_regressor = KerasRegressor(build_fn=build_lstm_model, epochs=50, batch_size=32, verbose=0)
simplernn_regressor = KerasRegressor(build_fn=build_simplernn_model, epochs=50, batch_size=32, verbose=0)
gru_regressor = KerasRegressor(build_fn=build_gru_model, epochs=50, batch_size=32, verbose=0)

# Define the hyperparameters to tune
param_grid = {
    'neurons': [50, 100],
    'dropout_rate': [0.2, 0.5]
}

# Create GridSearchCV objects for each model type
lstm_grid_search = GridSearchCV(estimator=lstm_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
simplernn_grid_search = GridSearchCV(estimator=simplernn_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
gru_grid_search = GridSearchCV(estimator=gru_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)

# Perform hyperparameter tuning for each model
lstm_grid_result = lstm_grid_search.fit(X_train, y_train)
simplernn_grid_result = simplernn_grid_search.fit(X_train, y_train)
gru_grid_result = gru_grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding mean test score for each model
print("LSTM Best Parameters: ", lstm_grid_result.best_params_)
print("LSTM Best Score: ", lstm_grid_result.best_score_)

print("SimpleRNN Best Parameters: ", simplernn_grid_result.best_params_)
print("SimpleRNN Best Score: ", simplernn_grid_result.best_score_)

print("GRU Best Parameters: ", gru_grid_result.best_params_)
print("GRU Best Score: ", gru_grid_result.best_score_)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D, Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional, ConvLSTM1D, ConvLSTM2D, ConvLSTM3D, RNN
from tensorflow.keras.layers import MultiHeadAttention, Attention, AdditiveAttention

# Define your hyperparameter space
num_epochs = [10, 20, 30]
batch_sizes = [16, 32, 64]
learning_rates = [0.001, 0.01, 0.1]

# Define your dataset and preprocessing steps here
# ...

# List of layers to iterate through
layers_to_test = [
    Conv1D,
    Conv2D,
    Conv3D,
    SeparableConv1D,
    SeparableConv2D,
    DepthwiseConv2D,
    Conv1DTranspose,
    Conv2DTranspose,
    Conv3DTranspose,
    MaxPooling1D,
    MaxPooling2D,
    MaxPooling3D,
    AveragePooling1D,
    AveragePooling2D,
    AveragePooling3D,
    GlobalMaxPooling1D,
    GlobalMaxPooling2D,
    GlobalMaxPooling3D,
    LSTM,
    GRU,
    SimpleRNN,
    TimeDistributed,
    Bidirectional,
    ConvLSTM1D,
    ConvLSTM2D,
    ConvLSTM3D,
    RNN,
    MultiHeadAttention,
    Attention,
    AdditiveAttention
]

# Iterate through combinations of layers and hyperparameters
for layer_type in layers_to_test:
    for epochs in num_epochs:
        for batch_size in batch_sizes:
            for learning_rate in learning_rates:
                model = Sequential()
                
                # Add the current layer to the model
                if layer_type in [Conv1D, Conv2D, Conv3D, SeparableConv1D, SeparableConv2D, DepthwiseConv2D,
                                  Conv1DTranspose, Conv2DTranspose, Conv3DTranspose]:
                    model.add(layer_type(64, (3, 3), activation='relu', input_shape=(input_shape)))
                elif layer_type in [MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D,
                                    AveragePooling3D, GlobalMaxPooling1D, GlobalMaxPooling2D, GlobalMaxPooling3D]:
                    model.add(layer_type(pool_size=(2, 2)))
                elif layer_type in [LSTM, GRU, SimpleRNN, TimeDistributed, Bidirectional, ConvLSTM1D, ConvLSTM2D,
                                    ConvLSTM3D, RNN]:
                    model.add(layer_type(64, activation='relu'))
                elif layer_type in [MultiHeadAttention, Attention, AdditiveAttention]:
                    model.add(layer_type())
                
                # Add more layers if needed
                # model.add(AdditionalLayer(...))

                # Compile the model with the current hyperparameters
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss='mean_squared_error',
                              metrics=['accuracy'])

                # Train the model on your dataset
                # model.fit(...)

                # Evaluate the model
                # loss, accuracy = model.evaluate(...)