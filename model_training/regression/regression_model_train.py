# %%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from data import preprocess_data
# Import all regression models
from regression_model_build import (
    build_LSTM_model,
    build_GRU_model,
    build_SimpleRNN_model,
    build_Conv1D_model,
    build_Conv1DPooling_model,
    build_Conv1D_LSTM_model,
    build_LSTM_CNN_Hybrid_model,
    build_Attention_LSTM_model,
    build_Transformer_model,
    build_BiLSTM_Attention_model,
    build_MultiStream_Hybrid_model,
    build_ResNet_model,
    build_TCN_model
)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# %% Configuration
# Select which model to use - change this to try different models
MODEL_TYPE = "LSTM"  # Options: LSTM, GRU, SimpleRNN, Conv1D, Conv1DPooling, Conv1D_LSTM, 
                    # LSTM_CNN_Hybrid, Attention_LSTM, Transformer, BiLSTM_Attention, 
                    # MultiStream_Hybrid, ResNet, TCN

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_model,
    "GRU": build_GRU_model,
    "SimpleRNN": build_SimpleRNN_model,
    "Conv1D": build_Conv1D_model,
    "Conv1DPooling": build_Conv1DPooling_model,
    "Conv1D_LSTM": build_Conv1D_LSTM_model,
    "LSTM_CNN_Hybrid": build_LSTM_CNN_Hybrid_model,
    "Attention_LSTM": build_Attention_LSTM_model,
    "Transformer": build_Transformer_model,
    "BiLSTM_Attention": build_BiLSTM_Attention_model,
    "MultiStream_Hybrid": build_MultiStream_Hybrid_model,
    "ResNet": build_ResNet_model,
    "TCN": build_TCN_model
}

# %% Load and preprocess data
print("Loading and preprocessing data...")
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_features.csv'
data = pd.read_csv(file_path, header=0)

# Optional: Use subset for faster testing
data = data.tail(1000)
print(f"Using {len(data)} rows of data")

# Preprocess the data
print("Preprocessing data...")
data = preprocess_data.clean_data(data)
# Split features and target
features = data.drop(columns=['Close'])  # All features except Close
target = data[['Close']]  # Target is 'Close'

# Configuration for window size and forecast horizon
n_in = 240   # Lookback window (number of timesteps)
n_out = 15   # Number of future timesteps to predict
n_features = features.shape[1]  # Number of feature columns

print(f"Features shape: {features.shape}, Target shape: {target.shape}")
print(f"Using lookback window of {n_in} timesteps and forecasting {n_out} timesteps ahead")
# Make sure data is numeric before creating sliding windows
print("Converting data to float32...")
features_values = features.values.astype(np.float32)

# Create sliding windows with numeric data
print("Creating sliding windows for features...")
features_array = sliding_window_view(features_values, n_in, axis=0)

# For regression, we want to predict future values
# Since we're predicting n_out future values, we need to align our targets
# We'll create sliding window of targets too, but with window size of n_out
# and offset from the end of each input window
print("Creating forecast targets...")
target_windows = []

# Create a window of n_out future prices for each input window
for i in range(len(features) - n_in - n_out + 1):
    target_window = target.values[i + n_in:i + n_in + n_out].astype(np.float32)
    target_windows.append(target_window)

target_array = np.array(target_windows)

# Ensure features_array matches target_array in length 
features_array = features_array[:len(target_array)]

# Now check for inf/nan values safely
print("Checking for invalid values...")
features_array = np.where(~np.isfinite(features_array), np.nan, features_array)
if np.isnan(features_array).sum() > 0:
    print(f"Warning: Found {np.isnan(features_array).sum()} NaN values in features. Replacing with column means.")
    # Reshape to 2D for easier preprocessing
    orig_shape = features_array.shape
    features_array_2d = features_array.reshape(-1, n_features)
    # Calculate column means (ignoring NaNs)
    col_means = np.nanmean(features_array_2d, axis=0)
    # Replace NaNs with column means
    for i in range(n_features):
        mask = np.isnan(features_array_2d[:, i])
        features_array_2d[mask, i] = col_means[i]
    # Reshape back to original shape
    features_array = features_array_2d.reshape(orig_shape)

target_array = np.where(np.isinf(target_array), np.nan, target_array)
if np.isnan(target_array).sum() > 0:
    print(f"Warning: Found {np.isnan(target_array).sum()} NaN values in targets. Replacing with means.")
    target_means = np.nanmean(target_array)
    target_array = np.nan_to_num(target_array, nan=target_means)

print(f"Features window shape: {features_array.shape}")
print(f"Target window shape: {target_array.shape}")

# Train-test split (maintain time sequence)
print("Splitting data into train and test sets...")
train_X, test_X, train_y, test_y = train_test_split(
    features_array, target_array, test_size=0.2, random_state=seed, shuffle=False
)

# Apply standardization to each feature across the time dimension
print("Standardizing features...")
# Reshape to 2D for scaling
train_X_2d = train_X.reshape(-1, n_features)
scaler_X = RobustScaler()
train_X_2d = scaler_X.fit_transform(train_X_2d)
train_X = train_X_2d.reshape(train_X.shape)

# Apply same scaling to test data
test_X_2d = test_X.reshape(-1, n_features)
test_X_2d = scaler_X.transform(test_X_2d)
test_X = test_X_2d.reshape(test_X.shape)


# Scale targets - important for regression
scaler_y = RobustScaler()
train_y_2d = train_y.reshape(-1, 1)
test_y_2d = test_y.reshape(-1, 1)

train_y_2d = scaler_y.fit_transform(train_y_2d)
test_y_2d = scaler_y.transform(test_y_2d)

# Reshape back to original shape
train_y = train_y_2d.reshape(train_y.shape)
test_y = test_y_2d.reshape(test_y.shape)

print(f"Train X shape: {train_X.shape}")
print(f"Test X shape: {test_X.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Test y shape: {test_y.shape}")

# %% Model training
print(f"Setting up {MODEL_TYPE} model for hyperparameter tuning...")

try:
    # Create the tuner using the selected model
    model_builder = MODEL_BUILDERS[MODEL_TYPE]
    
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp),
        objective='val_loss',
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        directory='models_dir',
        project_name=f'{MODEL_TYPE}_regression_tuning'
    )

    # Create callbacks for early stopping and learning rate adjustment
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Perform hyperparameter search
    print("Starting hyperparameter search...")
    tuner.search(
        train_X, train_y,
        epochs=10,  # Limited epochs for tuning
        batch_size=64,
        validation_data=(test_X, test_y),
        callbacks=callbacks
    )

    # Get the best hyperparameters and model
    print("Hyperparameter search complete. Getting best model...")
    best_hp = tuner.get_best_hyperparameters(1)[0]
    best_model = tuner.get_best_models(1)[0]

    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    for param in best_hp.values:
        print(f"- {param}: {best_hp.values[param]}")

    # Evaluate the model
    print("\nEvaluating best model on test data...")
    evaluation = best_model.evaluate(test_X, test_y, verbose=1)
    
    # Print metrics
    metrics = best_model.metrics_names
    print("\nTest Metrics:")
    for i, metric in enumerate(metrics):
        print(f"{metric}: {evaluation[i]:.4f}")

    # Train the final model using the best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    history = best_model.fit(
        train_X, train_y,
        epochs=50,
        batch_size=64,
        validation_data=(test_X, test_y),
        verbose=1,
        callbacks=callbacks
    )

    # Plot training history
    print("Plotting training history...")
    pyplot.figure(figsize=(15, 10))
    
    # Plot loss
    pyplot.subplot(2, 2, 1)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.title('Model Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    
    # Generate predictions
    print("Generating predictions on test data...")
    y_pred = best_model.predict(test_X)
    
    # Inverse transform predictions and actual values to original scale
    if len(y_pred.shape) > 2:  # Handle multi-step predictions
        y_pred_flat = y_pred.reshape(-1, 1)
        test_y_flat = test_y.reshape(-1, 1)
        
        y_pred_inv = scaler_y.inverse_transform(y_pred_flat).reshape(y_pred.shape)
        test_y_inv = scaler_y.inverse_transform(test_y_flat).reshape(test_y.shape)
    else:
        y_pred_inv = scaler_y.inverse_transform(y_pred)
        test_y_inv = scaler_y.inverse_transform(test_y)
    
    # Calculate regression metrics on original scale
    rmse_values = []
    mae_values = []
    r2_values = []
    
    # Calculate metrics for each forecast step
    for i in range(n_out):
        forecast_step_pred = y_pred_inv[:, i].flatten()
        forecast_step_true = test_y_inv[:, i].flatten()
        
        rmse = sqrt(mean_squared_error(forecast_step_true, forecast_step_pred))
        mae = mean_absolute_error(forecast_step_true, forecast_step_pred)
        r2 = r2_score(forecast_step_true, forecast_step_pred)
        
        rmse_values.append(rmse)
        mae_values.append(mae)
        r2_values.append(r2)
    
    # Overall metrics across all forecasts
    overall_rmse = sqrt(mean_squared_error(test_y_inv.flatten(), y_pred_inv.flatten()))
    overall_mae = mean_absolute_error(test_y_inv.flatten(), y_pred_inv.flatten())
    overall_r2 = r2_score(test_y_inv.flatten(), y_pred_inv.flatten())
    
    print(f"\nOverall Performance Metrics:")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    print(f"R²: {overall_r2:.4f}")
    
    # Plot metrics by forecast horizon
    pyplot.subplot(2, 2, 2)
    x_steps = np.arange(1, n_out + 1)
    pyplot.plot(x_steps, rmse_values, marker='o', label='RMSE')
    pyplot.plot(x_steps, mae_values, marker='s', label='MAE')
    pyplot.title('Error by Forecast Horizon')
    pyplot.xlabel('Steps Ahead')
    pyplot.ylabel('Error')
    pyplot.legend()
    pyplot.grid(True)
    
    # Plot R² by forecast horizon
    pyplot.subplot(2, 2, 3)
    pyplot.plot(x_steps, r2_values, marker='d', color='green')
    pyplot.title('R² by Forecast Horizon')
    pyplot.xlabel('Steps Ahead')
    pyplot.ylabel('R²')
    pyplot.grid(True)
    
    # Plot example predictions vs actual
    pyplot.subplot(2, 2, 4)
    # Select a random sample from the test set
    sample_idx = np.random.randint(0, len(test_X))
    
    # Plot actual vs predicted for the selected sample
    sample_pred = y_pred_inv[sample_idx]
    sample_true = test_y_inv[sample_idx]
    
    pyplot.plot(x_steps, sample_true, marker='o', label='Actual')
    pyplot.plot(x_steps, sample_pred, marker='x', label='Predicted')
    pyplot.title(f'Example Forecast (Sample #{sample_idx})')
    pyplot.xlabel('Steps Ahead')
    pyplot.ylabel('Price')
    pyplot.legend()
    pyplot.grid(True)
    
    pyplot.tight_layout()
    pyplot.savefig(f'{MODEL_TYPE}_regression_results.png')
    pyplot.show()
    
    # Save the model
    model_path = f'models/{MODEL_TYPE}_regressor.h5'
    best_model.save(model_path)
    print(f"Model saved to {model_path}")

except Exception as e:
    print(f"An error occurred during model training: {e}")
    import traceback
    traceback.print_exc()

print("Training process completed.")

#%%