#%% Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import sliding_window_view
import preprocess_data
# Import all classification models
from classification_model_build import (
    build_LSTM_classifier,
    build_GRU_classifier,
    build_Conv1D_classifier,
    build_Conv1D_LSTM_classifier,
    build_BiLSTM_Attention_classifier,
    build_Transformer_classifier,
    build_MultiStream_classifier,
    build_ResNet_classifier,
    build_TCN_classifier,
    build_CatBoostClassifier_model,
    build_LightGBMClassifier_model,
    build_XGBoostClassifier_model,
    build_RandomForestClassifier_model
)
# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

# Enable mixed precision training for A100 GPU
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_classifier,
    "GRU": build_GRU_classifier,
    "Conv1D": build_Conv1D_classifier,
    "Conv1D_LSTM": build_Conv1D_LSTM_classifier,
    # "BiLSTM_Attention": build_BiLSTM_Attention_classifier,
    # "Transformer": build_Transformer_classifier,
    # "MultiStream": build_MultiStream_classifier,
    # "ResNet": build_ResNet_classifier,
    # "TCN": build_TCN_classifier,
    # "CatBoost" : build_CatBoostClassifier_model,
    # "LightGBM" : build_LightGBMClassifier_model,
    # "XGBOOST" : build_XGBoostClassifier_model,
    # "RandomForest" : build_RandomForestClassifier_model

}

#%% SET VARIABLES
#----------------------- SET VARIABLES -----------------------#
target_col = 'short_signal'
raw_data = 'EURUSD_1min_sampled_features.csv'
# .txt file with Important Features list from Feature Importance Study
important_features_path = 'short_signal_important_features.txt'
# List of most important Lags from Feature Importance Study
lag_list = [70, 24, 10, 74, 39]
results_file_path = "sell_class_model_training_results.txt"
n_in = 240  # Number of past observations (lookback window)

# %% LOAD DATA
#----------------------- LOAD DATA -----------------------#
print("Loading Data...")
print(f"Loading data from {raw_data}...")
data = pd.read_csv(raw_data, header=0)
print("Preprocessing data...")
data = preprocess_data.clean_data(data)

#%% Transform Volume Based Features
#----------------------- Transform Volume Based Features -----------------------#
print("Transforming volume-based features...")
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

#%% ADD LAG FEATURES
#----------------------- ADD LAG FEATURES -----------------------#
print("Adding lag features...")

for lag in lag_list:
    data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
    
    # Also add lagged versions of key technical indicators
    for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
        if indicator in data.columns:
            data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)

# Add rolling stats on lagged features
data['target_lag_mean'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].mean(axis=1)
data['target_lag_std'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].std(axis=1)

#%% FEATURE SELECTION
#----------------------- FEATURE SELECTION -----------------------#
print("Performing feature selection...")
with open(important_features_path, 'r') as f:
    important_features = [line.strip() for line in f.readlines()]

# Add the lag features we created
important_features.extend([f'{target_col}_lag_{lag}' for lag in lag_list])
important_features.extend(['target_lag_mean', 'target_lag_std'])
all_cols = list(data.columns)
for col in all_cols:
    if '_log' in col or '_winsor' in col or '_rank' in col or 'lag_' in col:
        if col not in important_features:
            important_features.append(col)

# Filter the dataset to only include important features + target variables
keep_cols = important_features + ['long_signal', 'short_signal', 'close_position']
keep_cols = [col for col in keep_cols if col in data.columns]
data = data[keep_cols]

# Fill any missing values in lag features
data = data.fillna(method='bfill').fillna(method='ffill')

# Split features and target
columns_to_drop = ['long_signal', 'short_signal', 'close_position']
features = data.drop(columns=columns_to_drop)  # All features except targets
target = data[target_col]

# Ensure target is categorical (0 or 1)
target = target.astype(int)

#%% COMPUTE CLASS WEIGHT
#----------------------- COMPUTE CLASS WEIGHT -----------------------#
print("Getting Class Weights...")
classes = np.array([0, 1]) 
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=target.values.flatten()
)

class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
print(f"Computed class weights: {class_weight_dict}")

#%% BATCHING DATA
#----------------------- BATCHING DATA -----------------------#
print("Create Sliding Window...")

n_features = features.shape[1]  # Number of feature columns

print(f"Features shape after enhancement: {features.shape}, Target shape: {target.shape}")
print(f"Using lookback window of {n_in} timesteps")

# Apply sliding window to create time-series data
print("Creating sliding windows...")
features_array = sliding_window_view(features.values, n_in, axis=0)

features_array = sliding_window_view(features.values, n_in, axis=0)
target_array = target.values[n_in-1:]
features_array = features_array[:len(target_array)]

def create_time_series_generator(features, targets, window_size, batch_size=64, shuffle=False):
    """Memory-efficient generator for time series data"""
    data_len = len(features)
    indices = np.arange(window_size, data_len)
    
    while True:
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = np.array([features[i-window_size:i].values for i in batch_indices])
            batch_y = np.array(targets.iloc[batch_indices].values)
            
            yield batch_x, batch_y

def calculate_robust_batch_size_for_large_dataset(data_length, window_size, gpu_memory_gb=None):
    """
    Calculate an appropriate batch size for very large datasets
    
    Args:
        data_length: Length of dataset
        window_size: Size of lookback window
        gpu_memory_gb: GPU memory in GB (if known)
    
    Returns:
        batch_size: Calculated batch size
        steps_per_epoch: Number of steps per epoch
    """
    # Available samples after windowing
    available_samples = data_length - window_size
    
    if gpu_memory_gb is not None:
        # With known GPU memory, optimize for A100
        feature_count = n_features
        bytes_per_sample = feature_count * window_size * 4
        max_samples_in_memory = int(0.7 * gpu_memory_gb * 1e9 / bytes_per_sample)
        batch_size = min(1024, max_samples_in_memory)  # Increased for A100
    else:
        # Increase default for A100
        batch_size = 512  # Changed from 256
    
    # Calculate steps per epoch - for large datasets, we might not want to use all data in each epoch
    # Using approximately 100,000 samples per epoch is reasonable for most models
    target_samples_per_epoch = min(100000, available_samples)
    steps_per_epoch = target_samples_per_epoch // batch_size
    
    # Ensure at least 50 steps per epoch for stable training
    if steps_per_epoch < 50 and available_samples > 50 * batch_size:
        steps_per_epoch = 50
    
    print(f"Dataset size: {data_length:,} samples")
    print(f"Using batch size: {batch_size}, with {steps_per_epoch} steps per epoch")
    print(f"Each epoch will use {batch_size * steps_per_epoch:,} samples ({batch_size * steps_per_epoch / available_samples:.1%} of available data)")
    
    return batch_size, steps_per_epoch

# Define split point (80% train, 20% test)
split_idx = int(len(features) * 0.8)
print(f"Training set: {split_idx:,} samples, Test set: {len(features) - split_idx:,} samples")

# Calculate robust batch sizes for very large dataset
# If you know your GPU memory, specify it for better optimization
train_batch_size, train_steps = calculate_robust_batch_size_for_large_dataset(
    split_idx, n_in, gpu_memory_gb=40)
test_batch_size, val_steps = calculate_robust_batch_size_for_large_dataset(
    len(features) - split_idx, n_in, gpu_memory_gb=40)

# Create generators with calculated batch sizes
train_generator = create_time_series_generator(
    features[:split_idx], target[:split_idx], n_in, batch_size=train_batch_size, shuffle=False)
test_generator = create_time_series_generator(
    features[split_idx:], target[split_idx:], n_in, batch_size=test_batch_size, shuffle=False)

#%% HANDLE MISSING AND INF VALUES
#----------------------- HANDLE MISSING AND INF VALUES -----------------------#
print("Handling NaN and Inf values...")
import warnings
def handle_nan_inf_optimized(features_array, method='mean'):
    """
    Optimized function to handle NaN and Inf values in large arrays.
    
    Parameters:
    -----------
    features_array : numpy.ndarray
        Input array that may contain NaN or Inf values
    method : str
        Method to replace NaN values: 'mean', 'median', 'zero', 'forward_fill'
    
    Returns:
    --------
    numpy.ndarray
        Cleaned array with NaN/Inf values handled
    """
    print("Handling NaN and Inf values...")
    
    # Store original shape
    orig_shape = features_array.shape
    n_features = orig_shape[-1] if len(orig_shape) > 1 else 1
    
    # Quick check for any issues
    has_inf = np.isinf(features_array).any()
    has_nan = np.isnan(features_array).any()
    
    if not (has_inf or has_nan):
        print("No NaN or Inf values found.")
        return features_array
    
    # Work with a copy to avoid modifying original
    result = features_array.copy()
    
    # Handle Inf values first (replace with 0 or large finite values)
    if has_inf:
        inf_count = np.isinf(result).sum()
        print(f"Found {inf_count} Inf values. Replacing with 0.")
        result = np.where(np.isinf(result), 0, result)
    
    # Handle NaN values
    if has_nan:
        nan_count = np.isnan(result).sum()
        print(f"Warning: Found {nan_count} NaN values. Replacing with {method}.")
        
        if method == 'zero':
            # Fastest option - replace with zeros
            result = np.where(np.isnan(result), 0, result)
            
        elif method == 'mean':
            # Vectorized mean replacement
            if len(orig_shape) > 1:
                # Reshape to 2D for feature-wise operations
                result_2d = result.reshape(-1, n_features)
                
                # Calculate means ignoring NaNs (more efficient than nanmean for large arrays)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    col_means = np.nanmean(result_2d, axis=0)
                
                # Handle case where entire columns are NaN
                col_means = np.where(np.isnan(col_means), 0, col_means)
                
                # Vectorized replacement using broadcasting
                nan_mask = np.isnan(result_2d)
                result_2d[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
                
                # Reshape back
                result = result_2d.reshape(orig_shape)
            else:
                # 1D array case
                mean_val = np.nanmean(result)
                if np.isnan(mean_val):
                    mean_val = 0
                result = np.where(np.isnan(result), mean_val, result)
                
        elif method == 'median':
            # Similar to mean but using median
            if len(orig_shape) > 1:
                result_2d = result.reshape(-1, n_features)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    col_medians = np.nanmedian(result_2d, axis=0)
                col_medians = np.where(np.isnan(col_medians), 0, col_medians)
                
                nan_mask = np.isnan(result_2d)
                result_2d[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
                result = result_2d.reshape(orig_shape)
            else:
                median_val = np.nanmedian(result)
                if np.isnan(median_val):
                    median_val = 0
                result = np.where(np.isnan(result), median_val, result)
                
        elif method == 'forward_fill':
            # Forward fill for time series data
            if len(orig_shape) > 1:
                result_2d = result.reshape(-1, n_features)
                for i in range(n_features):
                    mask = np.isnan(result_2d[:, i])
                    if mask.any():
                        # Forward fill
                        valid_indices = np.where(~mask)[0]
                        if len(valid_indices) > 0:
                            result_2d[:, i] = np.interp(
                                np.arange(len(result_2d[:, i])),
                                valid_indices,
                                result_2d[valid_indices, i]
                            )
                        else:
                            result_2d[:, i] = 0
                result = result_2d.reshape(orig_shape)
            else:
                # Simple forward fill for 1D
                mask = np.isnan(result)
                if mask.any():
                    valid_indices = np.where(~mask)[0]
                    if len(valid_indices) > 0:
                        result = np.interp(np.arange(len(result)), valid_indices, result[valid_indices])
                    else:
                        result.fill(0)
    
    print(f"Processed array shape: {result.shape}")
    
    # Final verification
    remaining_nan = np.isnan(result).sum()
    remaining_inf = np.isinf(result).sum()
    
    if remaining_nan > 0 or remaining_inf > 0:
        print(f"Warning: {remaining_nan} NaN and {remaining_inf} Inf values still remain.")
    else:
        print("All NaN and Inf values successfully handled.")
    
    return result

features_array = handle_nan_inf_optimized(features_array, method='mean')
print(f"Window data shape: {features_array.shape}")
print(f"Target data shape: {target_array.shape}")

#%% SPLIT DATASET
#----------------------- SPLIT DATASET -----------------------#
# 7. Train-test split (maintain time sequence)
print("Splitting data into train and test sets...")
train_features, test_features, train_target, test_target = train_test_split(
    features, target, test_size=0.2, random_state=seed, shuffle=False)

#%% SCALE FEATURES
#----------------------- SCALE FEATURES -----------------------#
print("Scaling features...")

train_X = sliding_window_view(train_features.values, n_in, axis=0)
train_y = train_target.values[n_in-1:]
train_X = train_X[:len(train_y)]

# Create windows for test data
test_X = sliding_window_view(test_features.values, n_in, axis=0)
test_y = test_target.values[n_in-1:]
test_X = test_X[:len(test_y)]

# Now scale, fitting scaler ONLY on training data
scaler = RobustScaler(quantile_range=(10.0, 90.0))
train_X_2d = train_X.reshape(-1, n_features)
train_X_2d = scaler.fit_transform(train_X_2d)
train_X = train_X_2d.reshape(train_X.shape)

# Apply the already-fitted scaler to test data
test_X_2d = test_X.reshape(-1, n_features)
test_X_2d = scaler.transform(test_X_2d)
test_X = test_X_2d.reshape(test_X.shape)

# Check data stats after scaling
train_X_stats = {
    'min': np.min(train_X),
    'max': np.max(train_X),
    'mean': np.mean(train_X),
    'std': np.std(train_X)
}
print(f"Training data stats after scaling and capping: {train_X_stats}")

# Flatten target arrays
train_y = train_y.flatten()
test_y = test_y.flatten()

train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
train_dataset = train_dataset.batch(train_batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y))
test_dataset = test_dataset.batch(test_batch_size).prefetch(tf.data.AUTOTUNE)

#%% ADD FOCAL LOSS FOR NEURAL NETWORKS
#----------------------- ADD FOCAL LOSS FOR NEURAL NETWORKS -----------------------#
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        # Use a larger epsilon to prevent numerical issues
        epsilon = 1e-5
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Use a more stable implementation
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        
        # Take log and apply gamma factor
        loss = -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + epsilon)
        
        # Replace any NaN or inf with zeros
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        
        return tf.reduce_mean(loss)
    
    return focal_loss_fn

#%% CALLBACK METHODS
#----------------------- CALLBACK METHODS -----------------------#
nan_callback = tf.keras.callbacks.TerminateOnNaN()

class NaNSafetyCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor='loss', patience=3):
        super(NaNSafetyCallback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.nan_count = 0
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get(self.monitor)
        
        if loss is None:
            return
            
        if np.isnan(loss) or np.isinf(loss):
            self.nan_count += 1
            print(f"NaN/Inf detected in {self.monitor} at epoch {epoch+1}")
            
            if self.nan_count >= self.patience:
                print(f"Stopping training due to {self.nan_count} consecutive NaN/Inf values")
                self.model.stop_training = True
        else:
            # Reset counter if we get a valid loss
            self.nan_count = 0

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    min_delta=0.001  # Minimum change to count as improvement
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,  # Increased patience since we have warm-up
    min_lr=1e-6
)

def lr_schedule(epoch, lr):
    if epoch < 3:  # Warm-up phase
        return lr * 1.1  # Gradually increase LR during warm-up
    return lr
    
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

class EarlyStopper(tf.keras.callbacks.Callback):
    def __init__(self, baseline=0.8, min_epoch=5):
        super(EarlyStopper, self).__init__()
        self.baseline = baseline
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        # Only check after minimum epochs
        if epoch >= self.min_epoch:
            current = logs.get('val_accuracy')
            if current < self.baseline:
                print(f"\nStopping trial: val_accuracy {current} below threshold {self.baseline}")
                self.model.stop_training = True

early_stopper = EarlyStopper(baseline=0.65) 

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
     
def create_callbacks():
    """Create fresh callbacks for each trial"""
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule),
        EarlyStopper(baseline=0.65),
        NaNSafetyCallback(),
        GPUMemoryCallback()
    ]

#%% ROBUST PROCESSING
#----------------------- ROBUST PROCESSING -----------------------#
def robust_preprocessing(X_train, X_test, threshold=10.0):
    """
    Apply robust preprocessing to prevent NaNs during training for 3D time series data
    
    Args:
        X_train: Training data with shape (samples, timesteps, features)
        X_test: Test data with shape (samples, timesteps, features)
        threshold: Capping value for extreme values
        
    Returns:
        Processed X_train, X_test
    """
    # Get shapes
    n_samples_train, n_timesteps, n_features = X_train.shape
    
    # Replace NaNs with zeros
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=threshold, neginf=-threshold)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=threshold, neginf=-threshold)
    
    # Check if we have extreme values
    train_max = np.max(np.abs(X_train))
    if train_max > threshold:
        print(f"Warning: Extreme values detected ({train_max:.2f}). Clipping to ±{threshold}...")
        X_train = np.clip(X_train, -threshold, threshold)
        X_test = np.clip(X_test, -threshold, threshold)
    
    # Check for constant features across all timesteps
    # Reshape to 2D temporarily for easier analysis
    X_train_reshaped = X_train.reshape(-1, n_features)
    std_per_feature = np.std(X_train_reshaped, axis=0)
    constant_features = np.where(std_per_feature < 1e-10)[0]
    
    if len(constant_features) > 0:
        print(f"Warning: {len(constant_features)} constant features detected. Adding small noise...")
        for idx in constant_features:
            # Add tiny noise to each sample's feature, across all timesteps
            noise = np.random.normal(0, 1e-6, size=(n_samples_train, n_timesteps))
            # Ensure noise is properly broadcast to the right dimension (samples, timesteps)
            X_train[:, :, idx] += noise
    
    return X_train, X_test
# Use this instead of your current preprocessing
train_X, test_X = robust_preprocessing(train_X, test_X)

#%% PROGRESSIVE TRAINING
#----------------------- PROGRESSIVE TRAINING -----------------------#
def implement_progressive_training(model_builder, hp, train_X, train_y, test_X, test_y, 
                                   train_generator, test_generator, steps_per_epoch, validation_steps,
                                   callbacks, class_weight_dict, model_name):
    """
    Implement progressive training approach:
    1. First train on a smaller subset of data
    2. Then fine-tune on the full dataset
    
    Returns:
        Trained model
    """
    print(f"\n⚙️ Implementing progressive training for {model_name}...")
    
    # Phase 1: Initial training on smaller subset
    # Use 20% of training data for initial phase
    subset_size = len(train_X) // 5
    print(f"Phase 1: Training on {subset_size:,} samples ({subset_size/len(train_X):.1%} of training data)")
    
    # Build model with the given hyperparameters
    model = model_builder(hp)
    
    # Initial training phase - train on subset
    initial_history = model.fit(
        train_X[:subset_size], train_y[:subset_size],
        epochs=15, 
        batch_size=64,
        validation_data=(test_X, test_y),
        callbacks=create_callbacks(),
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print(f"Phase 1 completed. Initial validation metrics:")
    initial_eval = model.evaluate(test_X, test_y, verbose=0)
    for i, metric_name in enumerate(model.metrics_names):
        print(f"- {metric_name}: {initial_eval[i]:.4f}")
    
    # Phase 2: Full dataset fine-tuning
    print(f"\nPhase 2: Fine-tuning on full dataset ({len(train_X):,} samples)")
    
    # Reduce learning rate for fine-tuning phase
    K = tf.keras.backend
    current_lr = K.get_value(model.optimizer.learning_rate)
    K.set_value(model.optimizer.learning_rate, current_lr * 0.5)
    print(f"Reducing learning rate from {current_lr:.6f} to {current_lr * 0.5:.6f} for fine-tuning")
    
    # Fine-tuning on the full dataset - can use generator for memory efficiency
    if steps_per_epoch is not None and train_generator is not None:
        print("Using generator for fine-tuning phase")
        final_history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=35,
            validation_data=test_generator,
            validation_steps=validation_steps,
            callbacks=create_callbacks(),
            class_weight=class_weight_dict,
            verbose=1
        )
    else:
        # Fall back to regular training if generator not provided
        print("Using full dataset for fine-tuning phase")
        final_history = model.fit(
            train_X, train_y,
            epochs=35,
            batch_size=64,
            validation_data=(test_X, test_y),
            callbacks=create_callbacks(),
            class_weight=class_weight_dict,
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

#%% MODEL TRAINING
#----------------------- MODEL TRAINING -----------------------#
print(f"Setting up models for hyperparameter tuning...")

# Open the log file to store results
with open(results_file_path, "w") as log_file:
    log_file.write("Hyperparameter Tuning and Training Results\n")
    log_file.write("=" * 80 + "\n\n")

for model_name, model_builder in MODEL_BUILDERS.items():
    print(f"\n\n🚀 Training model: {model_name}")
    try:
        # Custom builder that incorporates focal loss
        def custom_model_builder(hp):
            # Special handling for models that need explicit input shape
            if model_name in ["MultiStream", "ResNet", "TCN"]:
                # These models need explicit input shape - (timesteps, features)
                input_shape = (n_in, train_X.shape[2])
                model = MODEL_BUILDERS[model_name](hp, input_shape=input_shape, num_classes=2)
            else:
                # Other models work with default parameters
                model = MODEL_BUILDERS[model_name](hp, num_classes=2)
            optimizer = tf.keras.optimizers.Adam(
                hp.Float("learning_rate", min_value=1e-6, max_value=5e-4, sampling="log"),
                clipvalue=1.0  # Clip gradients to avoid exploding values
            )
            model.compile(
                optimizer=optimizer,
                loss=focal_loss(
                    gamma=hp.Float("focal_gamma", min_value=1.0, max_value=3.0, step=0.5, default=2.0),
                    alpha=hp.Float("focal_alpha", min_value=0.1, max_value=0.9, step=0.1, default=0.25)
                ),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
                jit_compile=True  # Enable XLA compilation
            )
            return model

        train_X = np.nan_to_num(train_X, nan=0.0, posinf=0.0, neginf=0.0)
        test_X = np.nan_to_num(test_X, nan=0.0, posinf=0.0, neginf=0.0)

        # If values are extreme, consider capping them
        if train_X_stats['max'] > 10 or train_X_stats['min'] < -10:
            print("Warning: Extreme values detected in scaled training data. Capping values...")
            train_X = np.clip(train_X, -10, 10)
            test_X = np.clip(test_X, -10, 10)

        # 5. Updated callbacks with NaN detection
        # Define single consolidated callbacks list
        callbacks = [
            reduce_lr,
            nan_callback, 
            early_stopping_callback,
            lr_scheduler,
            early_stopper
        ]

        # Set up the hyperparameter tuner
        tuner = kt.BayesianOptimization(
            hypermodel=custom_model_builder,
            objective='val_loss',
            max_trials=200,
            directory='models',
            project_name=f'sell_trials_{model_name}',
            overwrite=True,
            executions_per_trial=1,
            max_consecutive_failed_trials=100,
            seed = 42
        )

        print("Starting hyperparameter search...")

        try:
            tuner.search(
                train_dataset,  # Replace train_X, train_y with dataset
                epochs=10,
                validation_data=test_dataset,  # Replace (test_X, test_y) with dataset
                callbacks=create_callbacks(),
                class_weight=class_weight_dict,
                verbose=1 
            )
        except Exception as e:
            print(f"Trial failed due to: {e}. Moving to next trial...")

        #----------------------- CROSS-VALIDATION -----------------------#
        # Implement cross-validation for model selection
        print("Selecting best model using custom criteria...")
        def time_series_cross_validate(tuner, X, y, n_folds=5, class_weights=None, callbacks=None):
            """
            Cross-validate with time-ordered splits, ensuring we have stable hyperparameters
            """
            print(f"Running {n_folds}-fold time series cross-validation...")
            
            # Get ALL hyperparameter configurations that completed successfully
            all_trials = tuner.oracle.get_best_trials(num_trials=100)  # Get many trials
            
            # Filter to only include trials that didn't result in NaN losses
            successful_trials = [
                trial for trial in all_trials 
                if trial.metrics.get_last_value('val_loss') is not None and 
                not np.isnan(trial.metrics.get_last_value('val_loss'))
            ]
            
            print(f"Found {len(successful_trials)} successful trials out of {len(all_trials)} total")
            
            # Need at least 3 successful hyperparameter sets
            if len(successful_trials) < 3:
                print("Not enough successful trials for cross-validation. Using best trial directly.")
                return tuner.get_best_hyperparameters(1)[0], []
            
            # Get top hyperparameter configurations from successful trials
            top_hps = [trial.hyperparameters for trial in successful_trials[:3]]
            
            # Setup TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=n_folds)
            
            cv_results = []
            
            # For each hyperparameter configuration
            for hp_idx, hp in enumerate(top_hps):
                print(f"\nValidating hyperparameter set {hp_idx+1}/{len(top_hps)}")
                
                fold_metrics = {
                    'val_loss': [],
                    'val_accuracy': [],
                    'val_precision': [],
                    'val_recall': [],
                    'val_auc': []
                }
                
                nan_count = 0
                
                # Run time series cross-validation
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    print(f"  Fold {fold+1}/{n_folds}")
                    
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            # Split data
                            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                            
                            # Build model with more stable options
                            model = tuner.hypermodel.build(hp)
                            optimizer = model.optimizer
                            # Add gradient clipping retrospectively
                            optimizer.clipnorm = 1.0
                            optimizer.clipvalue = 0.5
                            model.compile(optimizer=optimizer, loss=model.loss, metrics=model.metrics)
                            
                            # Train with early stopping and NaN detection
                            history = best_model.fit(
                                train_dataset,  # Use dataset instead of raw arrays
                                epochs=25,
                                # Remove batch_size parameter
                                validation_data=test_dataset,  # Use dataset instead of raw arrays
                                callbacks=create_callbacks(),
                                class_weight=class_weight_dict,
                                verbose=1
                            )
                            
                            # If we get here, training succeeded
                            break
                            
                        except Exception as e:
                            print(f"  Retry {retry+1}/{max_retries}: Error in fold {fold+1}: {e}")
                            if retry == max_retries - 1:
                                print("  All retries failed for this fold")
                                raise
                            # Clear session before retry
                            tf.keras.backend.clear_session()
                
                # Calculate mean and std of metrics across folds
                cv_summary = {}
                for metric, values in fold_metrics.items():
                    if values:  # Only if we have valid values
                        cv_summary[f'{metric}_mean'] = np.mean(values)
                        cv_summary[f'{metric}_std'] = np.std(values)
                
                cv_summary['nan_folds'] = nan_count
                cv_summary['valid_folds'] = n_folds - nan_count
                cv_summary['hyperparameters'] = hp
                
                # Only consider hyperparameter set if at least one fold was successful
                if nan_count < n_folds:  # Changed from 'n_folds // 2' to 'n_folds'
                    cv_results.append(cv_summary)
                    
                    val_loss_mean = cv_summary.get('val_loss_mean', 'N/A')
                    val_acc_mean = cv_summary.get('val_accuracy_mean', 'N/A')

                    # Use conditional formatting to prevent errors
                    val_loss_str = f"{val_loss_mean:.4f}" if isinstance(val_loss_mean, (int, float)) else str(val_loss_mean)
                    val_acc_str = f"{val_acc_mean:.4f}" if isinstance(val_acc_mean, (int, float)) else str(val_acc_mean)

                    print(f"  Results: val_loss_mean={val_loss_str}, val_acc_mean={val_acc_str}")
                else:
                    print(f"  ❌ All folds failed for this hyperparameter set - skipping")
            
            # Select the best hyperparameters based on cross-validation results
            if cv_results:
                # Sort by mean validation loss (ascending)
                cv_results.sort(key=lambda x: x.get('val_loss_mean', float('inf')))
                best_cv_result = cv_results[0]
                best_hp = best_cv_result['hyperparameters']
                
                print(f"\nBest cross-validated model:")
                print(f"- val_loss: {best_cv_result.get('val_loss_mean', 'N/A'):.4f} ± {best_cv_result.get('val_loss_std', 'N/A'):.4f}")
                print(f"- val_accuracy: {best_cv_result.get('val_accuracy_mean', 'N/A'):.4f} ± {best_cv_result.get('val_accuracy_std', 'N/A'):.4f}")
                print(f"- val_auc: {best_cv_result.get('val_auc_mean', 'N/A'):.4f} ± {best_cv_result.get('val_auc_std', 'N/A'):.4f}")
                
                return best_hp, cv_results
            else:
                # If all hyperparameter sets fail, just return the first one
                print("No valid hyperparameter sets found during cross-validation, using first set")
                return top_hps[0], []
            
        # Use this after hyperparameter tuning
        # After hyperparameter tuning
        best_hp, cv_results = time_series_cross_validate(
            tuner, 
            train_X, train_y, 
            n_folds=5, 
            class_weights=class_weight_dict,
            callbacks=create_callbacks()
        )

        # Make sure we have a hyperparameter set to work with, even if cross-validation fails
        if best_hp is None and len(tuner.get_best_hyperparameters(1)) > 0:
            best_hp = tuner.get_best_hyperparameters(1)[0]
            print("Using best hyperparameters from tuning instead of cross-validation")

        # Then build your final model with these hyperparameters
        if best_hp is not None:
            print("\nTraining final model with selected hyperparameters...")
            
            # Build a new model with the best hyperparameters
            best_model = tuner.hypermodel.build(best_hp)
            
            # Simple fit instead of progressive training to ensure completion
            history = best_model.fit(
                train_X, train_y,
                epochs=25,  # Reduced to ensure completion
                batch_size=64,
                validation_data=(test_X, test_y),
                callbacks=create_callbacks(),
                class_weight=class_weight_dict,
                verbose=1
            )
            
            # Save the trained model
            model_path = f'models/sell_models/{model_name}.h5'
            try:
                best_model.save(model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
            
            # Evaluate
            try:
                print(f"📊 Evaluating {model_name} model...")
                evaluation = best_model.evaluate(test_X, test_y, verbose=1)
                
                # Log results
                with open(results_file_path, "a") as log_file:
                    log_file.write(f"\n📌 Model: {model_name}\n")
                    log_file.write("=" * 40 + "\n")
                    log_file.write("Hyperparameters:\n")
                    for param in best_hp.values:
                        log_file.write(f"- {param}: {best_hp.values[param]}\n")

                    log_file.write("\nTest Metrics:\n")
                    for i, metric in enumerate(best_model.metrics_names):
                        log_file.write(f"{metric}: {evaluation[i]:.4f}\n")
                    
                    log_file.write("=" * 80 + "\n\n")
            except Exception as e:
                print(f"Error evaluating model: {e}")
                
        else:
            print("No hyperparameters available - skipping model training")
        
        print(f"📊 Evaluating {model_name} model...")
        evaluation = best_model.evaluate(test_X, test_y, verbose=1)

        # Log results
        with open(results_file_path, "a") as log_file:
            log_file.write(f"\n📌 Model: {model_name}\n")
            log_file.write("=" * 40 + "\n")
            for param in best_hp.values:
                log_file.write(f"- {param}: {best_hp.values[param]}\n")

            log_file.write("\nTest Metrics:\n")
            for i, metric in enumerate(best_model.metrics_names):
                log_file.write(f"{metric}: {evaluation[i]:.4f}\n")
            
            log_file.write("=" * 80 + "\n\n")

    except Exception as e:
        print(f"❌ Error training {model_name}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n✅ All models trained! Results saved to {results_file_path}")

#%%