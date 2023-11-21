#%%
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import ta 
from Strategies import call_Strategies

# Load your OHLCV and indicator/strategies datasets
# Assuming df_ohlc is the OHLCV dataset and df_indicators is the indicators/strategies dataset
# Make sure your datasets are appropriately preprocessed before loading
#%%
spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
df = pd.concat([indicators_df, all_signals_df], axis = 1)

# Assuming 'Close' is the target variable in df_ohlc
target_variable = 'Close'
print(indicators_df)
print(all_signals_df)
print(df)
print('indicators shape:',indicators_df.shape)
print('signals shape:',all_signals_df.shape)
print('df shape:',df.shape)

#%%
# Drop NaN values
df = df.dropna()

# Extract features and target variable
features = df[df.columns.difference([target_variable])].values  # Exclude 'Date' and target variable from features
target = df[target_variable]

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create sequences for LSTM
def create_sequences(data, sequence_length=10):
    sequences = []
    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# Define sequence length (you may adjust this based on your data)
sequence_length = 10

# Create sequences for features and target
X = create_sequences(scaled_features, sequence_length)
y = target.values[sequence_length:]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_train)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Mean Squared Error on Test Data: {loss}')

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and original target values to get back to the original scale
predictions = scaler.inverse_transform(predictions)
y_test_original_scale = scaler.inverse_transform(y_test.reshape(-1, 1))

# Now you can analyze and visualize the predictions and actual values
