# %%
import os
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view
from data import preprocess_data
import shap

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# ======================================
# MODEL BUILDERS IMPORT
# ======================================
from regression_model_build import (
    build_LSTM_model,
    build_GRU_model,
    build_SimpleRNN_model,
    build_Conv1D_model,
    build_Conv1DPooling_model,
    build_Conv1D_LSTM_model,
    build_LSTM_CNN_Hybrid_model,
    build_Transformer_model,
    build_MultiStream_Hybrid_model,
    build_ResNet_model,
    build_TCN_model
)

# ======================================
# CONFIGURATION MANAGEMENT
# ======================================
class ConfigManager:
    """
    Central configuration manager for regression model training
    """
    def __init__(self):
        # Data settings
        self.data_file = 'EURUSD_1min_sampled_features.csv'
        self.important_features_file = 'Close_important_features.txt'
        self.results_file = "regression_training_results.txt"
        
        # Model parameters
        self.lookback_window = 240
        self.forecast_horizon = 15
        self.test_size = 0.2
        self.random_seed = 42
        
        # Training parameters
        self.batch_size = 256
        self.max_epochs = 50
        self.early_stopping_patience = 10
        self.initial_learning_rate = 1e-3
        
        # Model selection
        self.selected_models = ['LSTM', "GRU", "SimpleRNN", "Conv1D", "TCN", "ResNet", "MultiStream_Hybrid", "Transformer", "LSTM_CNN_Hybrid", "Conv1D_LSTM", "Conv1DPooling"]
        
        # Data transformation
        self.use_log_transform = True
        self.use_lag_features = True
        self.lag_periods = [1, 2, 3, 4, 5]
            
    def save_config(self, filename="model_config.json"):
        """Save current configuration to JSON file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def load_config(self, filename="model_config.json"):
        """Load configuration from JSON file"""
        import json
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
                for key, value in config.items():
                    setattr(self, key, value)
        except FileNotFoundError:
            print(f"Config file {filename} not found. Using defaults.")

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_model,
    "GRU": build_GRU_model,
    "SimpleRNN": build_SimpleRNN_model,
    "Conv1D": build_Conv1D_model,
    "Conv1DPooling": build_Conv1DPooling_model,
    "Conv1D_LSTM": build_Conv1D_LSTM_model,
    "LSTM_CNN_Hybrid": build_LSTM_CNN_Hybrid_model,
    "Transformer": build_Transformer_model,
    "MultiStream_Hybrid": build_MultiStream_Hybrid_model,
    "ResNet": build_ResNet_model,
    "TCN": build_TCN_model
}

# Initialize config
config = ConfigManager()

# Make variables from config available globally
N_IN = config.lookback_window
N_OUT = config.forecast_horizon

# ======================================
# DATA LOADING AND PREPROCESSING
# ======================================
def load_and_preprocess_data(config):
    """
    Load and preprocess the data with feature engineering
    """
    print("Loading and preprocessing data...")
    data = pd.read_csv(config.data_file, header=0)
    print(f"Using {len(data)} rows of data")
    
    # Basic data cleaning
    data = preprocess_data.clean_data(data)
    
    # Transform volume-based features
    volume_cols = ['Volume']
    for col in volume_cols:
        if col in data.columns:
            # Log transform (handles high skewness)
            data[f'{col}_log'] = np.log1p(data[col])
            
            # Winsorize extreme values (cap at percentiles)
            q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
            data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
            
            # Rank transform (completely resistant to outliers)
            data[f'{col}_rank'] = data[col].rank(pct=True)
    
    # Add lag features
    target_col = 'Close'
    lag_list = config.lag_periods
    
    for lag in lag_list:
        data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Also add lagged versions of key technical indicators
        for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
            if indicator in data.columns:
                data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
    
    # Add rolling stats on lagged prices
    data['target_lag_mean'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].mean(axis=1)
    data['target_lag_std'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].std(axis=1)
    
    # Feature selection - load important features from file
    try:
        with open(config.important_features_file, 'r') as f:
            important_features = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(important_features)} important features from file")
    except FileNotFoundError:
        print(f"Important features file not found. Using all features.")
        important_features = []
    
    # Add the lag features and target to important features list
    important_features.extend([f'{target_col}_lag_{lag}' for lag in lag_list])
    important_features.extend(['target_lag_mean', 'target_lag_std'])
    if target_col not in important_features:
        important_features.append(target_col)
    
    # Add any transformed/new features
    all_cols = list(data.columns)
    for col in all_cols:
        if '_log' in col or '_winsor' in col or '_rank' in col or 'lag_' in col:
            if col not in important_features:
                important_features.append(col)
    
    # Filter features if important_features is not empty
    if important_features:
        keep_cols = [col for col in important_features if col in data.columns]
        data = data[keep_cols]
    
    # Fill missing values
    data = data.fillna(method='bfill').fillna(method='ffill')
    
    # Split features and target
    features = data.drop(columns=[target_col])
    target = data[[target_col]]
    
    return features, target, target_col

# ======================================
# TIME SERIES WINDOW CREATION 
# ======================================
def create_sliding_windows(features, target, n_in, n_out):
    """
    Create sliding windows for time series forecasting
    """
    print(f"Creating sliding windows with lookback={n_in}, horizon={n_out}")
    n_features = features.shape[1]
    
    # Convert data to float32
    features_values = features.values.astype(np.float32)
    
    # Create sliding windows for features
    features_array = sliding_window_view(features_values, n_in, axis=0)
    if len(features_array.shape) != 3:
        # Add additional check to ensure proper shape
        features_array = features_array.reshape(features_array.shape[0], n_in, n_features)
    # Create target windows (future values to predict)
    target_windows = []
    for i in range(len(features) - n_in - n_out + 1):
        target_window = target.values[i + n_in:i + n_in + n_out].astype(np.float32)
        target_windows.append(target_window)
    
    target_array = np.array(target_windows)
    
    # Ensure features_array matches target_array in length 
    features_array = features_array[:len(target_array)]
    
    # Handle missing and invalid values
    features_array = np.where(~np.isfinite(features_array), np.nan, features_array)
    if np.isnan(features_array).sum() > 0:
        print(f"Replacing {np.isnan(features_array).sum()} NaN values in features.")
        orig_shape = features_array.shape
        features_array_2d = features_array.reshape(-1, n_features)
        col_means = np.nanmean(features_array_2d, axis=0)
        for i in range(n_features):
            mask = np.isnan(features_array_2d[:, i])
            features_array_2d[mask, i] = col_means[i]
        features_array = features_array_2d.reshape(orig_shape)
    
    target_array = np.where(~np.isfinite(target_array), np.nan, target_array)
    if np.isnan(target_array).sum() > 0:
        print(f"Replacing {np.isnan(target_array).sum()} NaN values in target.")
        target_array = np.nan_to_num(target_array, nan=np.nanmean(target_array))
    
    print(f"Features shape: {features_array.shape}, Target shape: {target_array.shape}")
    return features_array, target_array

def create_windowed_time_series_generator(features, targets, window_size, batch_size=64, shuffle=False):
    """
    Memory-efficient generator for time series data
    Supports both 1-step and multi-step forecasting
    """
    # Convert inputs to numpy arrays for consistency
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(targets, pd.DataFrame):
        targets = targets.values
        
    data_len = len(features)
    valid_indices = np.arange(window_size, data_len-targets.shape[1]+1 if len(targets.shape) > 1 else data_len)
    
    if shuffle:
        np.random.shuffle(valid_indices)
    
    total_batches = int(np.ceil(len(valid_indices) / batch_size))
    
    while True:
        for batch_idx in range(total_batches):
            start_pos = batch_idx * batch_size
            batch_indices = valid_indices[start_pos:start_pos + batch_size]
            
            # Create batch input windows
            batch_x = np.array([features[i-window_size:i] for i in batch_indices])
            
            # Handle multi-step forecasting
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                # For multi-step targets, get the sequence of future values
                batch_y = np.array([targets[i:i+targets.shape[1]] for i in batch_indices])
            else:
                # For single-step targets, just get the next value
                batch_y = np.array([targets[i] for i in batch_indices])
                
            yield batch_x, batch_y

def calculate_steps_per_epoch(data_length, window_size, forecast_horizon, batch_size):
    """Calculate how many steps should be in each epoch"""
    valid_samples = data_length - window_size - forecast_horizon + 1
    return max(1, valid_samples // batch_size)

# ======================================
# DATA SCALING
# ======================================
def scale_data(train_X, test_X, train_y, test_y, n_features):
    """
    Scale the data using RobustScaler
    """
    print("Scaling features and targets...")
    
    # Scale features
    train_X_2d = train_X.reshape(-1, n_features)
    train_X_2d = np.nan_to_num(train_X_2d, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_X = RobustScaler(quantile_range=(10.0, 90.0))
    train_X_2d = scaler_X.fit_transform(train_X_2d)
    train_X_2d = np.clip(train_X_2d, -10, 10)  # Cap values to prevent instability
    train_X = train_X_2d.reshape(train_X.shape)
    
    # Scale test features
    test_X_2d = test_X.reshape(-1, n_features)
    test_X_2d = np.nan_to_num(test_X_2d, nan=0.0, posinf=0.0, neginf=0.0)
    test_X_2d = scaler_X.transform(test_X_2d)
    test_X_2d = np.clip(test_X_2d, -10, 10)
    test_X = test_X_2d.reshape(test_X.shape)
    
    # Scale targets
    scaler_y = RobustScaler(quantile_range=(10.0, 90.0))
    train_y_2d = train_y.reshape(-1, 1)
    test_y_2d = test_y.reshape(-1, 1)
    
    train_y_2d = scaler_y.fit_transform(train_y_2d)
    test_y_2d = scaler_y.transform(test_y_2d)
    
    # Reshape back to original shape
    train_y = train_y_2d.reshape(train_y.shape)
    test_y = test_y_2d.reshape(test_y.shape)
    
    data_stats = {
        'min': np.min(train_X),
        'max': np.max(train_X),
        'mean': np.mean(train_X),
        'std': np.std(train_X)
    }
    print(f"Data statistics after scaling: {data_stats}")
    
    return train_X, test_X, train_y, test_y, scaler_X, scaler_y

# ======================================
# CUSTOM LOSS FUNCTIONS & METRICS
# ======================================
def asymmetric_loss(y_true, y_pred, beta=1.0):
    """
    Asymmetric loss function that penalizes under-predictions more than over-predictions
    Especially useful for financial forecasting where missing upside is often worse
    
    Args:
        beta: Controls asymmetry. beta > 1 penalizes under-prediction more
    """
    error = y_true - y_pred
    under_forecast = tf.maximum(tf.zeros_like(error), error)
    over_forecast = tf.maximum(tf.zeros_like(error), -error)
    
    loss = tf.reduce_mean(beta * tf.square(under_forecast) + tf.square(over_forecast))
    return loss

def directional_accuracy_metric(y_true, y_pred):
    """
    Measures percentage of times the prediction direction matches actual direction
    Important for trading models where direction matters more than exact value
    """
    # For multi-step forecasts, compare adjacent steps
    true_direction = tf.sign(y_true[:, 1:] - y_true[:, :-1])
    pred_direction = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])
    
    # Compare directions
    correct_direction = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
    accuracy = tf.reduce_mean(correct_direction)
    return accuracy

# ======================================
# LEARNING RATE SCHEDULERS AND CALLBACKS
# ======================================
def cosine_annealing_warmup_schedule(epoch, lr, total_epochs=50, warmup_epochs=5, min_lr=1e-6):
    """
    Cosine annealing with warmup learning rate schedule
    """
    if epoch < warmup_epochs:
        # Linear warmup
        return lr * ((epoch + 1) / warmup_epochs)
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                logs['gpu_used_memory'] = mem_info['current'] / (1024**3)  # Convert to GB
                print(f"\nGPU Memory Used: {logs['gpu_used_memory']:.2f} GB")
        except Exception as e:
            print(f"Error monitoring GPU memory: {e}")

def get_training_callbacks():
    """
    Create callbacks for model training
    """
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
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_warmup_schedule(epoch, lr)
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        GPUMemoryCallback()
    ]
    return callbacks


# ======================================
# MODEL TRAINING UTILITIES
# ======================================
def implement_progressive_training(model_builder, hp, train_X, train_y, test_X, test_y, 
                                  callbacks, model_name):
    """
    Implement progressive training approach for regression with proper error handling:
    1. First train on a smaller subset of data
    2. Then fine-tune on the full dataset
    
    Returns:
        Trained model, training history or (None, None) if model building fails
    """
    print(f"\n⚙️ Implementing progressive training for {model_name}...")
    
    # Build model with the given hyperparameters
    try:
        model = model_builder(hp)
        if model is None:
            print(f"Error: Model builder returned None for {model_name}")
            return None, None
    except Exception as e:
        print(f"Error building model for {model_name}: {e}")
        return None, None
    
    # Phase 1: Initial training on smaller subset
    subset_size = len(train_X) // 5
    print(f"Phase 1: Training on {subset_size:,} samples ({subset_size/len(train_X):.1%} of training data)")
    
    try:
        # Initial training phase - train on subset
        initial_history = model.fit(
            train_X[:subset_size], train_y[:subset_size],
            epochs=15, 
            batch_size=64,
            validation_data=(test_X, test_y),
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Phase 1 completed. Initial validation metrics:")
        initial_eval = model.evaluate(test_X, test_y, verbose=0)
        for i, metric_name in enumerate(model.metrics_names):
            print(f"- {metric_name}: {initial_eval[i]:.4f}")
    except Exception as e:
        print(f"Error in Phase 1 training for {model_name}: {e}")
        return None, None
    
    # Phase 2: Full dataset fine-tuning
    try:
        print(f"\nPhase 2: Fine-tuning on full dataset ({len(train_X):,} samples)")
        
        # Reduce learning rate for fine-tuning phase
        try:
            K = tf.keras.backend
            # Get the current learning rate safely
            if hasattr(model.optimizer, 'learning_rate'):
                lr = model.optimizer.learning_rate
                # Check if learning rate is a tensor or a string/float
                if hasattr(lr, 'numpy'):
                    current_lr = float(lr.numpy())
                elif isinstance(lr, float):
                    current_lr = lr
                else:
                    # If it's a string or other type, create a new optimizer
                    print(f"Learning rate is of type {type(lr)}, creating new optimizer")
                    current_lr = 0.0001  # Default value
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr * 0.5),
                        loss=model.loss,
                        metrics=model.metrics
                    )
                    print(f"Set learning rate to {current_lr * 0.5:.6f}")
            else:
                print("Optimizer doesn't have a learning_rate attribute, using default")
                current_lr = 0.0001
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr * 0.5),
                    loss=model.loss,
                    metrics=model.metrics
                )
                print(f"Set learning rate to {current_lr * 0.5:.6f}")
        
        except Exception as e:
            print(f"Error adjusting learning rate: {e}")
            print("Continuing with original optimizer")
            
        # Fine-tuning on the full dataset
        final_history = model.fit(
            train_X, train_y,
            epochs=35,
            batch_size=64,
            validation_data=(test_X, test_y),
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Phase 2 completed. Final validation metrics:")
        final_eval = model.evaluate(test_X, test_y, verbose=0)
        for i, metric_name in enumerate(model.metrics_names):
            print(f"- {metric_name}: {final_eval[i]:.4f}")
        
        # Compare before and after fine-tuning
        print("\nImprovement from progressive training:")
        for i, metric_name in enumerate(model.metrics_names):
            diff = final_eval[i] - initial_eval[i]
            if 'loss' in metric_name:
                # For loss, lower is better
                print(f"- {metric_name}: {initial_eval[i]:.4f} → {final_eval[i]:.4f} ({diff:.4f})")
            else:
                # For accuracy, AUC, etc., higher is better
                print(f"- {metric_name}: {initial_eval[i]:.4f} → {final_eval[i]:.4f} (+{diff:.4f})")
        
        return model, final_history
    
    except Exception as e:
        print(f"Error in Phase 2 training for {model_name}: {e}")
        return None, None

def get_custom_model_builder(model_name, model_builder, n_out, train_X):
    def custom_model_builder(hp):
        try:
            # Print shape information for debugging
            print(f"Building {model_name} model with input shape: {train_X.shape}")
            
            # Get the actual builder function
            if isinstance(model_builder, str):
                if model_builder in MODEL_BUILDERS:
                    builder_func = MODEL_BUILDERS[model_builder]
                    print('BUILDER FUNCTION PRINT:', builder_func)
                else:
                    print(f"Unknown model type: {model_builder}")
                    return None
            else:
                builder_func = model_builder
                
            # Call the builder function with the required parameters
            try:
                # First try with train_X and n_out parameters
                base_model = builder_func(hp, train_X=train_X, n_out=n_out)
            except TypeError:
                try:
                    # If that fails, try without the named arguments
                    base_model = builder_func(hp, train_X, n_out)
                except TypeError:
                    try:
                        # If that fails too, try with just the hyperparameters
                        base_model = builder_func(hp)
                    except Exception as e:
                        print(f"Error creating {model_name} model: {e}")
                        return None
            
            # Check if model was successfully created
            if base_model is None:
                print(f"Model builder for {model_name} returned None")
                return None

            # Apply L2 regularization to prevent divergence (only for layers that support it)
            for layer in base_model.layers:
                if hasattr(layer, 'kernel_regularizer') and isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)

            # Choose loss function
            loss_type = hp.Choice("loss_type", ["huber", "mse", "asymmetric"], default="mse")
            
            if loss_type == "huber":
                loss_fn = tf.keras.losses.Huber()
            elif loss_type == "asymmetric":
                beta = hp.Float("asymmetric_beta", min_value=1.0, max_value=3.0, step=0.5, default=1.5)
                loss_fn = lambda y_true, y_pred: asymmetric_loss(y_true, y_pred, beta=beta)
            else:
                loss_fn = 'mse'
            
            # Lower learning rate for stability
            learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, 
                                    sampling="log", default=1e-4)
            
            # Safely compile the model with error handling
            try:
                base_model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=learning_rate,
                        clipnorm=1.0  # Prevent exploding gradients
                    ),
                    loss=loss_fn,
                    metrics=['mse', 'mae'],
                    jit_compile=True  # Enable XLA compilation
                )
                return base_model
            except Exception as e:
                print(f"Error compiling model for {model_name}: {e}")
                return None
                
        except Exception as e:
            print(f"Error in custom model builder for {model_name}: {e}")
            return None

    return custom_model_builder

def cross_validate_best_models(tuner, train_X, train_y, test_X, test_y, model_name, n_folds=5):
    """
    Cross-validate the top models from tuning to select the most robust one
    """
    print(f"Running {n_folds}-fold cross-validation on top models...")
    print(f"Input shapes - Train X: {train_X.shape}, Train y: {train_y.shape}")
    
    # Get top hyperparameter configurations
    try:
        top_hps = tuner.get_best_hyperparameters(3)
        if not top_hps:
            print("No valid hyperparameters found from tuning. Skipping cross-validation.")
            return None, None  # Return None to indicate failure
    except Exception as e:
        print(f"Error getting hyperparameters: {e}")
        return None, None  # Return None to indicate failure
    
    # Setup KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    cv_results = []
    
    # For each hyperparameter configuration
    for hp_idx, hp in enumerate(top_hps):
        print(f"\nValidating hyperparameter set {hp_idx+1}/{len(top_hps)}")
        
        fold_metrics = {
            'val_loss': [],
            'val_mse': [],
            'val_mae': []
        }
        
        # Build custom model builder for this configuration
        custom_builder = get_custom_model_builder(model_name, MODEL_BUILDERS[model_name], N_OUT, train_X)
        
        # Flag to track if all folds for this config were successful
        all_folds_successful = True
        
        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            try:
                # Split data
                X_train_fold, X_val_fold = train_X[train_idx], train_X[val_idx]
                y_train_fold, y_val_fold = train_y[train_idx], train_y[val_idx]
                print(f"  Fold training shapes: X={X_train_fold.shape}, y={y_train_fold.shape}")
            
                # Build model with these hyperparameters
                model = custom_builder(hp)
                if model is None:
                    print(f"  Model creation failed for fold {fold+1}")
                    all_folds_successful = False
                    break  # Exit fold loop if model creation fails
                    
                callbacks = get_training_callbacks()
                
                # Train model with reduced epochs for CV
                history = model.fit(
                    X_train_fold, y_train_fold,
                    epochs=10,  # Fewer epochs for CV
                    batch_size=64,
                    validation_data=(X_val_fold, y_val_fold),
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Get the best validation metrics
                if history.history.get('val_loss'):
                    best_epoch_idx = np.argmin(history.history['val_loss'])
                    fold_metrics['val_loss'].append(history.history['val_loss'][best_epoch_idx])
                    
                    if 'val_mse' in history.history:
                        fold_metrics['val_mse'].append(history.history['val_mse'][best_epoch_idx])
                    else:
                        fold_metrics['val_mse'].append(np.nan)
                        
                    if 'val_mae' in history.history:
                        fold_metrics['val_mae'].append(history.history['val_mae'][best_epoch_idx])
                    else:
                        fold_metrics['val_mae'].append(np.nan)
                else:
                    print(f"  No validation metrics found for fold {fold+1}")
                    all_folds_successful = False
                    break  # Exit fold loop if training fails
                    
            except Exception as e:
                print(f"  Error in fold {fold+1}: {str(e)}")
                all_folds_successful = False
                break  # Exit fold loop on exception
            finally:
                # Always clear session to free memory
                tf.keras.backend.clear_session()
        
        # Skip this hyperparameter set if any fold failed
        if not all_folds_successful or not fold_metrics['val_loss']:
            print(f"  Hyperparameter set {hp_idx+1} failed validation. Skipping.")
            continue
            
        # Calculate mean and std of metrics across folds
        cv_summary = {}
        for metric, values in fold_metrics.items():
            # Skip metrics with all NaN values
            if all(np.isnan(values)):
                cv_summary[f'{metric}_mean'] = np.nan
                cv_summary[f'{metric}_std'] = np.nan
            else:
                # Filter out NaN values
                filtered_values = [v for v in values if not np.isnan(v)]
                if filtered_values:
                    cv_summary[f'{metric}_mean'] = np.mean(filtered_values)
                    cv_summary[f'{metric}_std'] = np.std(filtered_values) if len(filtered_values) > 1 else 0.0
                else:
                    cv_summary[f'{metric}_mean'] = np.nan
                    cv_summary[f'{metric}_std'] = np.nan
        
        cv_summary['hyperparameters'] = hp
        cv_results.append(cv_summary)
        
        print(f"  Results: val_loss_mean={cv_summary.get('val_loss_mean', np.nan):.4f} ± {cv_summary.get('val_loss_std', np.nan):.4f}")
    
    # Select the best hyperparameters based on cross-validation results
    if cv_results:
        # Filter out results with NaN mean validation loss
        valid_results = [r for r in cv_results if not np.isnan(r.get('val_loss_mean', np.nan))]
        
        if valid_results:
            # Sort by mean validation loss (ascending)
            valid_results.sort(key=lambda x: x.get('val_loss_mean', float('inf')))
            best_cv_result = valid_results[0]
            best_hp = best_cv_result['hyperparameters']
            
            print(f"\nBest cross-validated model:")
            print(f"- val_loss: {best_cv_result.get('val_loss_mean', np.nan):.4f} ± {best_cv_result.get('val_loss_std', np.nan):.4f}")
            print(f"- val_mse: {best_cv_result.get('val_mse_mean', np.nan):.4f} ± {best_cv_result.get('val_mse_std', np.nan):.4f}")
            print(f"- val_mae: {best_cv_result.get('val_mae_mean', np.nan):.4f} ± {best_cv_result.get('val_mae_std', np.nan):.4f}")
            
            return best_hp, valid_results
    
    # If no valid results, return None instead of defaulting to first hyperparameter set
    print("No valid cross-validation results. Skipping this model.")
    return None, None
# ======================================
# MODEL EVALUATION UTILITIES
# ======================================
def evaluate_forecasts(y_pred, test_y, scaler_y, n_out):
    """
    Evaluate model forecasts with appropriate metrics
    Handles different output shapes and scaling
    """
    # Inverse transform predictions and actual values to original scale
    if len(test_y.shape) > 2:  
        test_y_flat = test_y.reshape(-1, test_y.shape[-1])  
        test_y_inv = scaler_y.inverse_transform(test_y_flat).reshape(test_y.shape)  
    else:
        test_y_inv = scaler_y.inverse_transform(test_y)

    if len(y_pred.shape) > 2:
        y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])  
        y_pred_inv = scaler_y.inverse_transform(y_pred_flat).reshape(y_pred.shape)
    else:
        y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Calculate metrics for each forecast step
    rmse_values = []
    mae_values = []
    r2_values = []
    
    print(f"Shape of test_y_inv: {test_y_inv.shape}")
    print(f"Shape of y_pred_inv: {y_pred_inv.shape}")
    
    # Handle different prediction shapes
    if len(test_y_inv.shape) == 3 and len(y_pred_inv.shape) == 2 and y_pred_inv.shape[1] == 1:
        # Single-step prediction vs multi-step true values
        print("Computing metrics using first forecast step only")
        test_y_first_step = test_y_inv[:, 0, 0]  # First step prediction of test set
        y_pred_flat = y_pred_inv[:, 0]           # First (and only) prediction step
        
        first_step_rmse = sqrt(mean_squared_error(test_y_first_step, y_pred_flat))
        first_step_mae = mean_absolute_error(test_y_first_step, y_pred_flat)
        first_step_r2 = r2_score(test_y_first_step, y_pred_flat)
        
        print(f"First step metrics - RMSE: {first_step_rmse:.4f}, MAE: {first_step_mae:.4f}, R²: {first_step_r2:.4f}")
        
        # Save overall metrics
        overall_rmse = first_step_rmse
        overall_mae = first_step_mae
        overall_r2 = first_step_r2
        
        # Compare each step with the repeated prediction
        all_true = []
        all_pred = []
        
        for i in range(min(n_out, test_y_inv.shape[1])):
            step_true = test_y_inv[:, i, 0]
            # Use the same prediction for all steps (since we only have one)
            step_pred = y_pred_inv[:, 0]
            
            step_rmse = sqrt(mean_squared_error(step_true, step_pred))
            step_mae = mean_absolute_error(step_true, step_pred)
            step_r2 = r2_score(step_true, step_pred)
            
            print(f"Step {i+1} vs Prediction - RMSE: {step_rmse:.4f}, MAE: {step_mae:.4f}, R²: {step_r2:.4f}")
            
            rmse_values.append(step_rmse)
            mae_values.append(step_mae)
            r2_values.append(step_r2)
            
            all_true.append(step_true)
            all_pred.append(step_pred)
        
        # Concatenate for overall metrics
            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            
            combined_rmse = sqrt(mean_squared_error(all_true, all_pred))
            combined_mae = mean_absolute_error(all_true, all_pred)
            combined_r2 = r2_score(all_true, all_pred)
            
            print(f"Combined metrics - RMSE: {combined_rmse:.4f}, MAE: {combined_mae:.4f}, R²: {combined_r2:.4f}")
            
        else:
            # Standard approach for matching shapes (multi-step prediction)
            for i in range(n_out):
                if len(test_y_inv.shape) == 3:
                    forecast_step_true = test_y_inv[:, i, 0]
                elif len(test_y_inv.shape) == 2:
                    forecast_step_true = test_y_inv[:, i]
                else:
                    forecast_step_true = test_y_inv
                
                if len(y_pred_inv.shape) == 2 and y_pred_inv.shape[1] > i:
                    forecast_step_pred = y_pred_inv[:, i]
                elif len(y_pred_inv.shape) == 3:
                    forecast_step_pred = y_pred_inv[:, i, 0]
                else:
                    forecast_step_pred = y_pred_inv
                
                rmse = sqrt(mean_squared_error(forecast_step_true, forecast_step_pred))
                mae = mean_absolute_error(forecast_step_true, forecast_step_pred)
                r2 = r2_score(forecast_step_true, forecast_step_pred)
                
                rmse_values.append(rmse)
                mae_values.append(mae)
                r2_values.append(r2)
                
                print(f"Step {i+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Try to compute overall metrics by flattening
            try:
                test_y_flat = test_y_inv.flatten()
                y_pred_flat = y_pred_inv.flatten()
                
                if len(test_y_flat) == len(y_pred_flat):
                    overall_rmse = sqrt(mean_squared_error(test_y_flat, y_pred_flat))
                    overall_mae = mean_absolute_error(test_y_flat, y_pred_flat)
                    overall_r2 = r2_score(test_y_flat, y_pred_flat)
                else:
                    print(f"WARNING: Flattened shapes don't match: {len(test_y_flat)} vs {len(y_pred_flat)}")
                    # Use average of step metrics as fallback
                    overall_rmse = np.mean(rmse_values)
                    overall_mae = np.mean(mae_values)
                    overall_r2 = np.mean(r2_values)
            except Exception as e:
                print(f"Error computing overall metrics: {str(e)}")
                # Use average of step metrics as fallback
                overall_rmse = np.mean(rmse_values)
                overall_mae = np.mean(mae_values)
                overall_r2 = np.mean(r2_values)
        
        evaluation_results = {
            'overall_rmse': overall_rmse,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'rmse_values': rmse_values,
            'mae_values': mae_values,
            'r2_values': r2_values
        }
        
        return evaluation_results

def plot_forecast_results(history, evaluation_results, sample_true, sample_pred, model_name, n_out):
    """
    Plot training history and forecast evaluation results
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot training history
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics by forecast horizon
    ax2 = fig.add_subplot(2, 2, 2)
    x_steps = np.arange(1, n_out + 1)
    ax2.plot(x_steps, evaluation_results['rmse_values'], marker='o', label='RMSE')
    ax2.plot(x_steps, evaluation_results['mae_values'], marker='s', label='MAE')
    ax2.set_title('Error by Forecast Horizon')
    ax2.set_xlabel('Steps Ahead')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)
    
    # Plot R² by forecast horizon
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(x_steps, evaluation_results['r2_values'], marker='d', color='green')
    ax3.set_title('R² by Forecast Horizon')
    ax3.set_xlabel('Steps Ahead')
    ax3.set_ylabel('R²')
    ax3.grid(True)
    
    # Plot example prediction
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(x_steps, sample_true, marker='o', label='Actual')
    ax4.plot(x_steps, sample_pred, marker='x', label='Predicted')
    ax4.set_title('Example Forecast')
    ax4.set_xlabel('Steps Ahead')
    ax4.set_ylabel('Price')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_regression_results.png')
    plt.close()

# ======================================
# MAIN TRAINING PIPELINE
# ======================================
def train_regression_models(config):
    """
    Main function to train multiple regression models
    """
    # Set up logging
    log_file_path = config.results_file
    with open(log_file_path, "w") as log_file:
        log_file.write("Multi-step Regression Models Training Results\n")
        log_file.write("=" * 80 + "\n\n")
    
    # Step 1: Load and preprocess data
    features, target, target_col = load_and_preprocess_data(config)
    
    # Step 2: Create sliding windows
    features_array, target_array = create_sliding_windows(
        features, target, config.lookback_window, config.forecast_horizon
    )
    
    # Step 3: Split into train/test sets
    train_X, test_X, train_y, test_y = train_test_split(
        features_array, target_array, 
        test_size=config.test_size, 
        random_state=config.random_seed, 
        shuffle=False
    )
    
    # Step 4: Scale the data
    n_features = features.shape[1]
    train_X, test_X, train_y, test_y, scaler_X, scaler_y = scale_data(
        train_X, test_X, train_y, test_y, n_features
    )

    # Convert to optimized tf.data.Dataset (add this code)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    train_dataset = train_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
    test_dataset = test_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
        
    # Get list of models to train
    if config.selected_models:
        model_list = config.selected_models
    else:
        model_list = list(MODEL_BUILDERS.keys())
    
    # Step 5: Iterate through models
    for model_name in model_list:
        print(f"\n\n{'='*50}")
        print(f"🚀 Training model: {model_name}")
        print(f"{'='*50}")
        print('BUILDER FUNCTION PRINT:', MODEL_BUILDERS[model_name])
        
        try:
            # Log model training start
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n\n## Model: {model_name}\n")
                log_file.write("=" * 50 + "\n")
            
            # Get model builder and callbacks
            model_builder = MODEL_BUILDERS[model_name]
            callbacks = get_training_callbacks()
            custom_builder = get_custom_model_builder(model_name, model_builder, config.forecast_horizon, train_X=train_X)
            
            # Step 6: Hyperparameter tuning
            print("\nStarting hyperparameter search...")
            tuner = kt.Hyperband(
                hypermodel=lambda hp: custom_builder(hp),
                objective='val_loss',
                max_epochs=30,
                factor=3,
                hyperband_iterations=1,
                directory='models_dir',
                project_name=f'{model_name}_regression_tuning',
                overwrite=True
            )
            
            tuner_success = True
            try:
                tuner.search(
                    train_dataset,  # Replace train_X, train_y with train_dataset
                    epochs=10,
                    validation_data=test_dataset,  # Replace (test_X, test_y) with test_dataset
                    callbacks=callbacks
                )
            except Exception as e:
                print(f"Warning: Hyperparameter search error: {e}")
                print(f"Skipping model {model_name} due to hyperparameter search failure")
                tuner_success = False
                import traceback
                traceback.print_exc()
                
                # Log the error
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n❌ Error in hyperparameter search for {model_name}: {str(e)}\n")
                    log_file.write("Skipping this model.\n")
                    log_file.write(f"Detailed error: {str(e)}")
                    log_file.write(f"Error type: {type(e)}")
                
                # Skip to the next model
                continue
            
            # Step 7: Cross-validation only if hyperparameter tuning succeeded
            if tuner_success:
                best_hp, cv_results = cross_validate_best_models(
                    tuner, train_X, train_y, test_X, test_y, model_name
                )
                
                # Skip the rest of the training if cross-validation failed
                if best_hp is None:
                    print(f"No valid hyperparameters found for {model_name}. Moving to next model.")
                    
                    # Log the failure
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"\n❌ Cross-validation failed for {model_name}. No valid hyperparameters found.\n")
                        log_file.write("Skipping this model.\n")
                    
                    continue
                
                # Step 8: Train final model
                print("\nTraining final model with cross-validated hyperparameters...")
                best_model, history = implement_progressive_training(
                    model_builder=lambda hp: custom_builder(best_hp),
                    hp=best_hp,
                    train_X=train_X,  # Keep these parameters
                    train_y=train_y, 
                    test_X=test_X, 
                    test_y=test_y,
                    train_dataset=train_dataset,  # Add new parameters
                    test_dataset=test_dataset,
                    callbacks=callbacks,
                    model_name=model_name
                )
                
                # Print best hyperparameters
                print("\nBest Hyperparameters:")
                for param in best_hp.values:
                    print(f"- {param}: {best_hp.values[param]}")
                    
                # Step 9: Generate predictions and evaluate
                y_pred = best_model.predict(test_X)
                evaluation_results = evaluate_forecasts(y_pred, test_y, scaler_y, config.forecast_horizon)
                
                print(f"\nOverall Performance Metrics:")
                print(f"RMSE: {evaluation_results['overall_rmse']:.4f}")
                print(f"MAE: {evaluation_results['overall_mae']:.4f}")
                print(f"R²: {evaluation_results['overall_r2']:.4f}")
            
            # Step 10: Plot results
            # Select a random sample for visualization
            sample_idx = np.random.randint(0, len(test_X))
            
            # Handle different shapes for visualization
            if len(test_y.shape) == 3 and test_y.shape[2] == 1:
                sample_true = test_y[sample_idx, :, 0]
            else:
                sample_true = test_y[sample_idx]
                
            if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
                # Single-step prediction - repeat for visualization
                sample_pred = np.full(config.forecast_horizon, y_pred[sample_idx, 0])
            else:
                sample_pred = y_pred[sample_idx]
            
            # Inverse transform the sample for plotting
            sample_true = scaler_y.inverse_transform(sample_true.reshape(-1, 1)).flatten()
            sample_pred = scaler_y.inverse_transform(sample_pred.reshape(-1, 1)).flatten()
            
            plot_forecast_results(
                history, 
                evaluation_results, 
                sample_true, 
                sample_pred, 
                model_name, 
                config.forecast_horizon
            )
            
            # Step 11: Save the model
            model_path = f'models/{model_name}_regressor.keras'
            best_model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Log results
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n📌 Model: {model_name}\n")
                log_file.write("=" * 40 + "\n")
                
                # Log hyperparameters
                log_file.write("\nHyperparameters:\n")
                for param in best_hp.values:
                    log_file.write(f"- {param}: {best_hp.values[param]}\n")
                
                # Log metrics
                log_file.write("\nPerformance Metrics:\n")
                log_file.write(f"- RMSE: {evaluation_results['overall_rmse']:.4f}\n")
                log_file.write(f"- MAE: {evaluation_results['overall_mae']:.4f}\n")
                log_file.write(f"- R²: {evaluation_results['overall_r2']:.4f}\n")
                
                # Log forecast horizon metrics
                log_file.write("\nForecast Horizon Metrics:\n")
                for i, step in enumerate(range(1, config.forecast_horizon + 1)):
                    log_file.write(f"- Step {step} - RMSE: {evaluation_results['rmse_values'][i]:.4f}, ")
                    log_file.write(f"MAE: {evaluation_results['mae_values'][i]:.4f}, ")
                    log_file.write(f"R²: {evaluation_results['r2_values'][i]:.4f}\n")
                
                log_file.write("\n" + "=" * 80 + "\n")
        
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Log the error
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n❌ Error training {model_name}: {str(e)}\n")
                log_file.write("=" * 80 + "\n")
    
    print("\n🎉 All models training completed!")
    print(f"Results saved to {log_file_path}")

#%%
# ======================================
# MAIN EXECUTION
# ======================================
if __name__ == "__main__":
    # Better GPU detection and configuration
    try:
        # First check if TensorFlow can see any GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"TensorFlow detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu}")
                
            # Try to enable memory growth to avoid allocating all memory at once
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("Memory growth enabled for all GPUs")
            except RuntimeError as e:
                print(f"Error enabling memory growth: {e}")
                print("Will use default memory allocation")
                
        else:
            print("No GPUs detected by TensorFlow. Checking system...")
            
            # Check if CUDA is available via environment
            import os
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"  # Enable detailed TF logs
            
            # Try setting visible devices explicitly
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU if available
            
            # Check for NVIDIA GPUs using system commands
            try:
                import subprocess
                gpu_info = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if gpu_info.returncode == 0:
                    print("NVIDIA GPU detected in system:")
                    print(gpu_info.stdout.decode())
                    
                    # Force TensorFlow to check for GPUs again
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        print(f"Successfully detected {len(gpus)} GPU(s) after environment setup")
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    else:
                        # Check for common issues
                        print("GPU detected by nvidia-smi but not by TensorFlow. Checking for issues:")
                        
                        # Check CUDA and cuDNN installation
                        print("\nChecking TensorFlow CUDA configuration:")
                        print(f"TensorFlow version: {tf.__version__}")
                        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
                        print(f"Is GPU available: {tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else 'Function deprecated'}")
                        
                        # Print GPU device info if available
                        with tf.device('/CPU:0'):
                            print("\nTrying to create a simple tensor on CPU:")
                            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                            print(f"Tensor device: {a.device}")
                        
                        try:
                            with tf.device('/GPU:0'):
                                print("\nTrying to create a simple tensor on GPU:")
                                b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                                print(f"Tensor device: {b.device}")
                        except RuntimeError as e:
                            print(f"Error creating tensor on GPU: {e}")
                
                else:
                    print("No NVIDIA GPU detected with nvidia-smi")
                    print("Error output:", gpu_info.stderr.decode())
            except Exception as e:
                print(f"Error checking GPU with nvidia-smi: {e}")
            
            print("\nFalling back to CPU execution...")
            # Configure TensorFlow for optimal CPU usage
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            print("Optimized TensorFlow for CPU operation")
    
    except Exception as e:
        print(f"Error during GPU detection and configuration: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to default CPU configuration")
    
    # Verify what device TensorFlow will use
    print("\nFinal device configuration check:")
    print(f"Available devices: {tf.config.list_physical_devices()}")
    
    # Uncomment to load custom config
    # config.load_config("regression_config.json")
    
    # Start training
    train_regression_models(config)

#%%