import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Assuming you have a 1D dataset of size 6000
# Replace this with your actual dataset
dataset = np.random.rand(6000)

# Define parameters
input_size = 1
batch_size = 1
timesteps_per_batch = 30
validation_steps = 15

# Create LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(timesteps_per_batch, input_size)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Create ModelCheckpoint callback
checkpoint_path = 'recursive_checkpoint_model.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_best_only=True)

# Train the model recursively
for i in range(len(dataset) - timesteps_per_batch - validation_steps + 1):
    # Prepare batch data
    batch = dataset[i:i + timesteps_per_batch + validation_steps]

    # Split into input (X) and output (y)
    X, y = batch[:timesteps_per_batch], batch[timesteps_per_batch:]

    # Reshape input for LSTM (batch_size, timesteps, input_size)
    X = X.reshape((batch_size, timesteps_per_batch, input_size))

    # Train the model
    model.fit(X, y, epochs=1, batch_size=batch_size, validation_split=0.5, callbacks=[checkpoint_callback])

# Load the final model from the last checkpoint
model.load_weights(checkpoint_path)