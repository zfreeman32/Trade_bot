# %%
import os
import sys

def setup_environment():
    """Setup environment variables before TensorFlow import"""
    # Suppress TensorFlow warnings initially
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # CUDA configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    
    # GPU memory management
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    
    # Set library paths (adjust paths based on your system)
    cuda_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda-12.4/lib64', 
        '/usr/local/cuda-11.8/lib64',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    existing_paths = [path for path in cuda_paths if os.path.exists(path)]
    if existing_paths:
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = ':'.join(existing_paths)
        if current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path + ':' + current_ld_path
        else:
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
        print(f"Set LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

# Setup environment before ANY TensorFlow imports
setup_environment()

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split, KFold
from sklearn.preprocessing import RobustScaler
import keras_tuner as kt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import sliding_window_view
import preprocess_data

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def configure_gpu():
    """
    Configure GPU with comprehensive error handling and fallback
    MUST be called before any TensorFlow operations
    """
    print("🔧 Configuring GPU...")
    
    try:
        # Enable detailed logging temporarily
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        
        # Check for GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ Detected {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
            
            try:
                # Configure each GPU with proper memory management
                for gpu in gpus:
                    # Enable memory growth to prevent OOM errors
                    tf.config.experimental.set_memory_growth(gpu, True)

                print("✅ GPU memory growth enabled")
                
                # Set visible devices
                tf.config.experimental.set_visible_devices(gpus, 'GPU')
                
                # Test GPU availability with a small tensor
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0]], dtype=tf.float32)
                    result = tf.reduce_sum(test_tensor)
                    print(f"✅ GPU test successful. Result device: {result.device}")
                
                print("✅ GPU configuration completed")
                
                return True, len(gpus)
                
            except RuntimeError as e:
                print(f"⚠️  GPU configuration error: {e}")
                if "memory growth" in str(e).lower():
                    print("Continuing with default memory settings...")
                    return True, len(gpus)
                else:
                    print("Attempting CPU fallback...")
                    return False, 0
                    
        else:
            print("❌ No GPUs detected by TensorFlow")
            print("Checking possible causes:")
            print("- CUDA drivers: Run 'nvidia-smi' to verify")
            print("- TensorFlow version: Try 'pip install tensorflow-gpu'")
            print("- Environment: Check CUDA_VISIBLE_DEVICES")
            return False, 0
            
    except Exception as e:
        print(f"❌ GPU configuration error: {e}")
        print("Will attempt CPU fallback...")
        return False, 0
    
    finally:
        # Restore warning level
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def configure_cpu_optimal():
    """Configure TensorFlow for optimal CPU performance"""
    print("🔧 Configuring for CPU optimization...")
    
    try:
        # Set thread counts for CPU optimization
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
        
        # Disable GPU (force CPU usage)
        tf.config.set_visible_devices([], 'GPU')
        
        print("✅ CPU optimization configured")
        return True
        
    except RuntimeError as e:
        if "cannot be modified after initialization" in str(e):
            print("⚠️  TensorFlow already initialized. Cannot modify threading settings.")
            print("Restart Python session for optimal CPU configuration.")
        else:
            print(f"❌ CPU configuration error: {e}")
        return False

def setup_tensorflow():
    """Setup TensorFlow with GPU/CPU configuration"""
    print("🚀 Setting up TensorFlow...")
    
    # Try GPU configuration first
    gpu_success, num_gpus = configure_gpu()

    if gpu_success and num_gpus >= 1:
        print(f"✅ Using GPU acceleration with {num_gpus} GPU(s)")
        
        # Enable mixed precision for Tensor Cores
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
        
        return 'GPU', num_gpus
    else:
        print("⚠️  GPU setup failed. Configuring for CPU...")
        cpu_success = configure_cpu_optimal()
        if cpu_success:
            print("✅ Using optimized CPU configuration")
            return 'CPU', 0
        else:
            print("⚠️  Using default TensorFlow configuration")
            return 'DEFAULT', 0

# Configure TensorFlow before importing other modules
device_type, device_count = setup_tensorflow()

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

# ======================================
# CONFIGURATION MANAGEMENT
# ======================================
class ClassificationConfigManager:
    """
    Central configuration manager for classification model training
    """
    def __init__(self):
        # Data settings
        self.data_file = 'long_signal_dataset.csv'
        self.results_file = "buy_class_model_training_results.txt"
        
        # Model parameters
        self.target_col = 'long_signal'
        self.lookback_window = 120  # REDUCED from 240 to save memory
        self.test_size = 0.2
        self.random_seed = 42
        
        # Memory optimization settings
        self.max_samples = 500000   # REDUCED from 100000 to further limit dataset size
        self.chunk_size = 10000     # REDUCED chunk size
        self.use_streaming = True  # Use tf.data for memory efficiency
        
        # Training parameters
        self.batch_size = 512      # INCREASED back to 512 for fewer steps
        self.max_epochs = 25
        self.steps_per_epoch = 50  # LIMIT steps per epoch for faster training
        self.validation_steps = 25  # LIMIT validation steps
        self.early_stopping_patience = 5
        self.initial_learning_rate = 1e-4
        
        # Model selection
        self.selected_models = ['LSTM', 'GRU', 'Conv1D', 'Conv1D_LSTM']
        
        # Data transformation
        self.use_lag_features = True
        self.lag_periods = [61, 93, 64, 60, 77]
        
        # Hyperparameter tuning
        self.max_trials = 50       # REDUCED from 200
        self.max_consecutive_failed_trials = 20
            
    def save_config(self, filename="classification_config.json"):
        """Save current configuration to JSON file"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def load_config(self, filename="classification_config.json"):
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

# Initialize config
config = ClassificationConfigManager()

# Make variables from config available globally
n_in = config.lookback_window

# ======================================
# DATA LOADING AND PREPROCESSING
# ======================================
def load_and_preprocess_data(config):
    """
    Load and preprocess the data with feature engineering for classification
    Memory-optimized version for large datasets
    """
    print("Loading and preprocessing data...")
    data = pd.read_csv(config.data_file, header=0)
    print(f"Original dataset size: {len(data)} rows")
    
    # Sample data if too large to prevent memory issues
    if len(data) > config.max_samples:
        print(f"Dataset too large ({len(data)} rows). Sampling {config.max_samples} rows...")
        # Use stratified sampling to maintain class balance
        target_col_temp = data[config.target_col]
        
        _, data = train_test_split(
            data, 
            test_size=config.max_samples / len(data),
            stratify=target_col_temp,
            random_state=config.random_seed
        )
        print(f"Sampled dataset size: {len(data)} rows")
    
    # Basic data cleaning
    data = preprocess_data.clean_data(data)
    
    # Transform volume-based features (only if Volume column exists)
    volume_cols = ['Volume']
    for col in volume_cols:
        if col in data.columns:
            print(f"Transforming volume column: {col}")
            # Log transform (handles high skewness)
            data[f'{col}_log'] = np.log1p(data[col])
            
            # Winsorize extreme values (cap at percentiles)
            q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
            data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
            
            # Rank transform (completely resistant to outliers)
            data[f'{col}_rank'] = data[col].rank(pct=True)
    
    # Add lag features
    print("Adding lag features...")
    lag_list = config.lag_periods
    
    for lag in lag_list:
        data[f'{config.target_col}_lag_{lag}'] = data[config.target_col].shift(lag)
        
        # Also add lagged versions of key technical indicators
        for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
            if indicator in data.columns:
                data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
    
    # Add rolling stats on lagged features
    lag_cols = [f'{config.target_col}_lag_{lag}' for lag in lag_list if f'{config.target_col}_lag_{lag}' in data.columns]
    if lag_cols:
        data['target_lag_mean'] = data[lag_cols].mean(axis=1)
        data['target_lag_std'] = data[lag_cols].std(axis=1)
    
    # Since dataset has already been through feature importance study, 
    # we'll use all available features except target columns and time columns
    exclude_columns = [
        'long_signal', 'short_signal', 'close_position',  # Target columns
        'Date', 'Time', 'datetime', 'Open', 'High', 'Low', 'Close'  # Time and price columns
    ]
    
    # Get all feature columns (everything except excluded columns)
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    print(f"Using {len(feature_columns)} features from the pre-processed dataset")
    print(f"Excluded columns: {[col for col in exclude_columns if col in data.columns]}")
    
    # Fill missing values
    data = data.bfill().ffill()
    
    # Split features and target
    features = data[feature_columns]  # All features except excluded columns
    target = data[config.target_col]
    
    # Ensure target is categorical (0 or 1)
    target = target.astype(int)
    
    # Validate that all feature columns are numeric
    non_numeric_cols = []
    for col in features.columns:
        if not pd.api.types.is_numeric_dtype(features[col]):
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"Warning: Found non-numeric columns in features: {non_numeric_cols}")
        print("Attempting to convert to numeric or remove...")
        
        for col in non_numeric_cols:
            try:
                # Try to convert to numeric
                features[col] = pd.to_numeric(features[col], errors='coerce')
                print(f"Successfully converted {col} to numeric")
            except:
                print(f"Removing non-numeric column: {col}")
                features = features.drop(columns=[col])
    
    print(f"Final dataset shape - Features: {features.shape}, Target: {target.shape}")
    print(f"Memory usage: Features: {features.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    return features, target

# ======================================
# TIME SERIES WINDOW CREATION 
# ======================================
def create_sliding_windows_classification_chunked(features, target, n_in, chunk_size=10000):
    """
    Create sliding windows for time series classification using chunked processing
    to avoid memory issues with large datasets
    """
    print(f"Creating sliding windows with lookback={n_in} using chunked processing")
    n_features = features.shape[1]
    total_samples = len(features)
    
    print(f"Processing {total_samples} samples in chunks of {chunk_size}")
    
    # Calculate total windows needed
    total_windows = total_samples - n_in + 1
    if total_windows <= 0:
        raise ValueError(f"Not enough data: need at least {n_in} samples, got {total_samples}")
    
    print(f"Will create {total_windows} windows")
    
    # Convert data to float32 for memory efficiency
    features_values = features.values.astype(np.float32)
    target_values = target.values.astype(np.int32)
    
    # Process in chunks to avoid memory overflow
    feature_chunks = []
    target_chunks = []
    
    # Process data in overlapping chunks to maintain sequence continuity
    for start_idx in range(0, total_samples - n_in + 1, chunk_size):
        # Ensure we have enough data for the window
        end_idx = min(start_idx + chunk_size + n_in - 1, total_samples)
        
        print(f"Processing chunk {start_idx} to {end_idx} ({end_idx - start_idx} samples)")
        
        # Get chunk data
        chunk_features = features_values[start_idx:end_idx]
        chunk_targets = target_values[start_idx:end_idx]
        
        # Create windows for this chunk
        if len(chunk_features) >= n_in:
            chunk_windows = sliding_window_view(chunk_features, window_shape=n_in, axis=0)
            chunk_windows = np.transpose(chunk_windows, (0, 2, 1))
            
            # Corresponding targets (current time step prediction)
            chunk_target_windows = chunk_targets[n_in-1:n_in-1+len(chunk_windows)]
            
            # Only take the windows that don't overlap with next chunk
            if start_idx + chunk_size < total_samples - n_in + 1:
                take_windows = min(chunk_size, len(chunk_windows))
                chunk_windows = chunk_windows[:take_windows]
                chunk_target_windows = chunk_target_windows[:take_windows]
            
            feature_chunks.append(chunk_windows)
            target_chunks.append(chunk_target_windows)
    
    # Concatenate all chunks
    print("Concatenating chunks...")
    features_array = np.concatenate(feature_chunks, axis=0)
    target_array = np.concatenate(target_chunks, axis=0)
    
    # Handle NaNs in features
    nan_count = np.isnan(features_array).sum()
    if nan_count > 0:
        print(f"Replacing {nan_count} NaNs in features.")
        # Process in chunks to avoid memory issues
        for i in range(0, len(features_array), chunk_size):
            end_i = min(i + chunk_size, len(features_array))
            chunk = features_array[i:end_i]
            
            original_shape = chunk.shape
            chunk_2d = chunk.reshape(-1, n_features)
            
            # Replace NaNs with column means
            for j in range(n_features):
                col_data = chunk_2d[:, j]
                if np.isnan(col_data).any():
                    col_mean = np.nanmean(col_data)
                    chunk_2d[np.isnan(col_data), j] = col_mean
            
            features_array[i:end_i] = chunk_2d.reshape(original_shape)
    
    print(f"Final shapes — Features: {features_array.shape}, Target: {target_array.shape}")
    print(f"Memory usage: Features: {features_array.nbytes / 1e6:.1f} MB, Target: {target_array.nbytes / 1e6:.1f} MB")
    
    return features_array, target_array

# ======================================
# DATA SCALING
# ======================================
def scale_data_classification_chunked(train_X, test_X, n_features, chunk_size=10000):
    """
    Scale the data using RobustScaler for classification with chunked processing
    to avoid memory issues with large datasets
    """
    print("Scaling features using chunked processing...")
    print(f"Input shapes - Train: {train_X.shape}, Test: {test_X.shape}")
    print(f"Estimated memory usage: {(train_X.nbytes + test_X.nbytes) / 1e9:.2f} GB")
    
    # Create and fit the scaler using chunks of training data
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    
    print("Fitting scaler on training data...")
    # Fit scaler on chunks to avoid memory issues
    for i in range(0, len(train_X), chunk_size):
        end_i = min(i + chunk_size, len(train_X))
        chunk = train_X[i:end_i]
        
        # Reshape chunk to 2D for scaling
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Partial fit the scaler
        if i == 0:
            scaler.fit(chunk_2d)
        else:
            # For RobustScaler, we need to recompute on all data seen so far
            # This is a limitation, so we'll fit on the first large chunk only
            pass
    
    print("Transforming training data...")
    # Transform training data in chunks
    for i in range(0, len(train_X), chunk_size):
        end_i = min(i + chunk_size, len(train_X))
        chunk = train_X[i:end_i]
        
        # Reshape to 2D
        original_shape = chunk.shape
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transform
        chunk_2d = scaler.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)  # Cap values to prevent instability
        
        # Reshape back and store in-place
        train_X[i:end_i] = chunk_2d.reshape(original_shape)
    
    print("Transforming test data...")
    # Transform test data in chunks
    for i in range(0, len(test_X), chunk_size):
        end_i = min(i + chunk_size, len(test_X))
        chunk = test_X[i:end_i]
        
        # Reshape to 2D
        original_shape = chunk.shape
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Transform
        chunk_2d = scaler.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)
        
        # Reshape back and store in-place
        test_X[i:end_i] = chunk_2d.reshape(original_shape)
    
    data_stats = {
        'min': np.min(train_X),
        'max': np.max(train_X),
        'mean': np.mean(train_X),
        'std': np.std(train_X)
    }
    print(f"Data statistics after scaling: {data_stats}")
    
    return train_X, test_X, scaler

# ======================================
# FOCAL LOSS FOR NEURAL NETWORKS
# ======================================
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

# ======================================
# CALLBACKS AND LEARNING RATE SCHEDULERS
# ======================================
def cosine_annealing_warmup_schedule(epoch, lr, total_epochs=50, warmup_epochs=3, min_lr=1e-6):
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

class EarlyStopper(tf.keras.callbacks.Callback):
    def __init__(self, baseline=0.65, min_epoch=5):
        super(EarlyStopper, self).__init__()
        self.baseline = baseline
        self.min_epoch = min_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        # Only check after minimum epochs
        if epoch >= self.min_epoch:
            current = logs.get('val_accuracy')
            if current and current < self.baseline:
                print(f"\nStopping trial: val_accuracy {current} below threshold {self.baseline}")
                self.model.stop_training = True

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
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_warmup_schedule(epoch, lr)
        ),
        EarlyStopper(baseline=0.65),
        NaNSafetyCallback(),
        GPUMemoryCallback()
    ]

# ======================================
# ROBUST PREPROCESSING
# ======================================
def robust_preprocessing(X_train, X_test, threshold=10.0):
    """
    Apply robust preprocessing to prevent NaNs during training for 3D time series data
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
    X_train_reshaped = X_train.reshape(-1, n_features)
    std_per_feature = np.std(X_train_reshaped, axis=0)
    constant_features = np.where(std_per_feature < 1e-10)[0]
    
    if len(constant_features) > 0:
        print(f"Warning: {len(constant_features)} constant features detected. Adding small noise...")
        for idx in constant_features:
            noise = np.random.normal(0, 1e-6, size=(n_samples_train, n_timesteps))
            X_train[:, :, idx] += noise
    
    return X_train, X_test

# ======================================
# MODEL TRAINING UTILITIES
# ======================================
def get_custom_model_builder(model_name, model_builder, train_X):
    def custom_model_builder(hp):
        try:
            # Special handling for models that need explicit input shape
            if model_name in ["MultiStream", "ResNet", "TCN"]:
                # These models need explicit input shape - (timesteps, features)
                input_shape = (n_in, train_X.shape[2])
                model = model_builder(hp, input_shape=input_shape, num_classes=2)
            else:
                # Other models work with default parameters
                model = model_builder(hp, num_classes=2)
                
            if model is None:
                print(f"Model builder for {model_name} returned None")
                return None

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
        except Exception as e:
            print(f"Error in custom model builder for {model_name}: {e}")
            return None

    return custom_model_builder

def implement_progressive_training(model_builder, hp, train_X, train_y, test_X, test_y, 
                                   callbacks, class_weight_dict, model_name):
    """
    Implement progressive training approach for classification with proper error handling
    """
    print(f"\n⚙️ Implementing progressive training for {model_name}...")
    
    try:
        # Build model with the given hyperparameters
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
        initial_history = model.fit(
            train_X[:subset_size], train_y[:subset_size],
            epochs=15, 
            batch_size=64,
            validation_data=(test_X, test_y),
            callbacks=callbacks,
            class_weight=class_weight_dict,
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
            if hasattr(model.optimizer, 'learning_rate'):
                lr = model.optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    current_lr = float(lr.numpy())
                elif isinstance(lr, float):
                    current_lr = lr
                else:
                    current_lr = 0.0001
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr * 0.5),
                        loss=model.loss,
                        metrics=model.metrics
                    )
                    print(f"Set learning rate to {current_lr * 0.5:.6f}")
            else:
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
            class_weight=class_weight_dict,
            verbose=1
        )
        
        print(f"Phase 2 completed. Final validation metrics:")
        final_eval = model.evaluate(test_X, test_y, verbose=0)
        for i, metric_name in enumerate(model.metrics_names):
            print(f"- {metric_name}: {final_eval[i]:.4f}")
        
        return model, final_history
    
    except Exception as e:
        print(f"Error in Phase 2 training for {model_name}: {e}")
        return None, None

def cross_validate_best_models(tuner, train_X, train_y, test_X, test_y, model_name, class_weight_dict, n_folds=5):
    """
    Cross-validate the top models from tuning to select the most robust one for classification
    """
    print(f"Running {n_folds}-fold time series cross-validation...")
    
    # Get ALL hyperparameter configurations that completed successfully
    try:
        all_trials = tuner.oracle.get_best_trials(num_trials=100)
        
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
            best_hps = tuner.get_best_hyperparameters(1)
            return best_hps[0] if best_hps else None, []
        
        # Get top hyperparameter configurations from successful trials
        top_hps = [trial.hyperparameters for trial in successful_trials[:3]]
        
    except Exception as e:
        print(f"Error getting hyperparameters: {e}")
        return None, []
    
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
        all_folds_successful = True
        
        # Run time series cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(train_X)):
            print(f"  Fold {fold+1}/{n_folds}")
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Split data
                    X_train_fold, X_val_fold = train_X[train_idx], train_X[val_idx]
                    y_train_fold, y_val_fold = train_y[train_idx], train_y[val_idx]
                    
                    # Get custom model builder
                    custom_builder = get_custom_model_builder(model_name, MODEL_BUILDERS[model_name], train_X)
                    
                    # Build model with more stable options
                    model = custom_builder(hp)
                    if model is None:
                        print(f"  Model creation failed for fold {fold+1}")
                        all_folds_successful = False
                        break
                    
                    # Train with early stopping and NaN detection
                    history = model.fit(
                        X_train_fold, y_train_fold,
                        epochs=25,
                        batch_size=64,
                        validation_data=(X_val_fold, y_val_fold),
                        callbacks=create_callbacks(),
                        class_weight=class_weight_dict,
                        verbose=0
                    )
                    
                    # If we get here, training succeeded
                    if history.history.get('val_loss'):
                        best_epoch_idx = np.argmin(history.history['val_loss'])
                        for metric in fold_metrics.keys():
                            if metric in history.history:
                                fold_metrics[metric].append(history.history[metric][best_epoch_idx])
                    
                    break
                    
                except Exception as e:
                    print(f"  Retry {retry+1}/{max_retries}: Error in fold {fold+1}: {e}")
                    if retry == max_retries - 1:
                        all_folds_successful = False
                        nan_count += 1
                        break
                    tf.keras.backend.clear_session()
            
            if not all_folds_successful:
                break
        
        # Only consider hyperparameter set if at least half the folds were successful
        if nan_count < n_folds // 2:
            cv_summary = {}
            for metric, values in fold_metrics.items():
                if values:
                    cv_summary[f'{metric}_mean'] = np.mean(values)
                    cv_summary[f'{metric}_std'] = np.std(values)
            
            cv_summary['nan_folds'] = nan_count
            cv_summary['valid_folds'] = n_folds - nan_count
            cv_summary['hyperparameters'] = hp
            
            cv_results.append(cv_summary)
            
            val_loss_mean = cv_summary.get('val_loss_mean', 'N/A')
            val_acc_mean = cv_summary.get('val_accuracy_mean', 'N/A')
            print(f"  Results: val_loss_mean={val_loss_mean:.4f}, val_acc_mean={val_acc_mean:.4f}")
        else:
            print(f"  ❌ Too many failed folds for this hyperparameter set - skipping")
    
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
        print("No valid hyperparameter sets found during cross-validation")
        return None, []

# ======================================
# MAIN TRAINING PIPELINE
# ======================================
def train_classification_models(config):
    """
    Main function to train multiple classification models with memory optimization
    """
    try:
        import psutil
    except ImportError:
        print("psutil not available, memory monitoring disabled")
        psutil = None
        
    import gc
    
    def print_memory_usage(step_name):
        """Print current memory usage"""
        if psutil:
            memory = psutil.virtual_memory()
            print(f"[{step_name}] Memory usage: {memory.percent:.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
        else:
            print(f"[{step_name}] Memory monitoring not available")
            
        # Check GPU memory if available
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                gpu_used = mem_info['current'] / (1024**3)
                gpu_total = mem_info['peak'] / (1024**3)
                print(f"[{step_name}] GPU memory: {gpu_used:.1f}/{gpu_total:.1f} GB")
        except:
            pass
    
    print_memory_usage("Start")
    
    # Set up logging
    log_file_path = config.results_file
    with open(log_file_path, "w") as log_file:
        log_file.write("Classification Models Training Results\n")
        log_file.write("=" * 80 + "\n\n")
    
    # Step 1: Load and preprocess data
    features, target = load_and_preprocess_data(config)
    print_memory_usage("After data loading")
    
    # Step 2: Create sliding windows using chunked processing
    features_array, target_array = create_sliding_windows_classification_chunked(
        features, target, config.lookback_window, chunk_size=config.chunk_size
    )
    print_memory_usage("After window creation")
    
    # Free original data to save memory
    del features, target
    gc.collect()
    print_memory_usage("After cleanup")
    
    # Step 3: Compute class weights
    print("Computing class weights...")
    classes = np.array([0, 1]) 
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target_array.flatten()
    )
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
    print(f"Computed class weights: {class_weight_dict}")
    
    # Step 4: Split into train/test sets
    print("Splitting data...")
    train_X, test_X, train_y, test_y = train_test_split(
        features_array, target_array, 
        test_size=config.test_size, 
        random_state=config.random_seed, 
        shuffle=False
    )
    print_memory_usage("After train/test split")
    
    # Free original arrays to save memory
    del features_array, target_array
    gc.collect()
    print_memory_usage("After split cleanup")
    
    # Step 5: Scale the data using chunked processing
    n_features = train_X.shape[2]  # Last dimension is features
    train_X, test_X, scaler = scale_data_classification_chunked(
        train_X, test_X, n_features, chunk_size=config.chunk_size
    )
    print_memory_usage("After scaling")
    
    # Step 6: Apply robust preprocessing
    train_X, test_X = robust_preprocessing(train_X, test_X)
    print_memory_usage("After preprocessing")
    
    # Flatten target arrays
    train_y = train_y.flatten()
    test_y = test_y.flatten()
    
    # Step 7: Create streaming datasets for memory efficiency
    print("Creating streaming datasets...")
    
    def create_streaming_dataset(X, y, batch_size, shuffle=False):
        """Create memory-efficient streaming dataset"""
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            # Use smaller buffer for shuffling to save memory
            buffer_size = min(5000, len(X))
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_dataset = create_streaming_dataset(train_X, train_y, config.batch_size, shuffle=True)
    test_dataset = create_streaming_dataset(test_X, test_y, config.batch_size, shuffle=False)
    
    # Calculate actual steps per epoch
    full_train_steps = max(1, len(train_X) // config.batch_size)
    full_val_steps = max(1, len(test_X) // config.batch_size)
    
    train_steps_per_epoch = min(config.steps_per_epoch, full_train_steps)
    val_steps_per_epoch = min(config.validation_steps, full_val_steps)
    
    print(f"Dataset sizes - Train: {len(train_X):,}, Test: {len(test_X):,}")
    print(f"Batch size: {config.batch_size}")
    print(f"Full training steps available: {full_train_steps}")
    print(f"Limited training steps per epoch: {train_steps_per_epoch}")
    print(f"Full validation steps available: {full_val_steps}")
    print(f"Limited validation steps per epoch: {val_steps_per_epoch}")
    print(f"Samples per training epoch: {train_steps_per_epoch * config.batch_size:,}")
    print(f"Samples per validation epoch: {val_steps_per_epoch * config.batch_size:,}")
    
    if train_steps_per_epoch < full_train_steps:
        reduction_pct = (1 - train_steps_per_epoch / full_train_steps) * 100
        print(f"🚀 Training time reduced by {reduction_pct:.1f}% due to step limiting")
    
    print_memory_usage("After dataset creation")
    
    # Get list of models to train
    if config.selected_models:
        model_list = config.selected_models
    else:
        model_list = list(MODEL_BUILDERS.keys())
    
    # Step 8: Iterate through models
    for model_name in model_list:
        print(f"\n\n{'='*50}")
        print(f"🚀 Training model: {model_name}")
        print(f"{'='*50}")
        print_memory_usage(f"Start {model_name}")
        
        try:
            # Clear any existing models from memory
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory_usage(f"After session clear - {model_name}")
            
            # Log model training start
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n\n## Model: {model_name}\n")
                log_file.write("=" * 50 + "\n")
            
            # Get model builder and callbacks
            model_builder = MODEL_BUILDERS[model_name]
            callbacks = create_callbacks()
            custom_builder = get_custom_model_builder(model_name, model_builder, train_X)
            
            # Step 9: Hyperparameter tuning with reduced trials for memory efficiency
            print("\nStarting hyperparameter search...")
            print(f"Using {train_steps_per_epoch} steps per epoch (instead of {full_train_steps} full steps)")
            if train_steps_per_epoch < full_train_steps:
                reduction_pct = (1 - train_steps_per_epoch / full_train_steps) * 100
                print(f"This reduces training time by {reduction_pct:.1f}%")
            tuner = kt.BayesianOptimization(
                hypermodel=custom_builder,
                objective='val_loss',
                max_trials=config.max_trials,  # Reduced from original
                directory='models',
                project_name=f'buy_trials_{model_name}',
                overwrite=True,
                executions_per_trial=1,
                max_consecutive_failed_trials=config.max_consecutive_failed_trials,
                seed=config.random_seed
            )

            tuner_success = True
            try:
                tuner.search(
                    train_dataset,
                    epochs=5,  # Reduced epochs for faster tuning
                    steps_per_epoch=train_steps_per_epoch,  # Limit steps per epoch
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,   # Limit validation steps
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1 
                )
                print_memory_usage(f"After hyperparameter search - {model_name}")
            except Exception as e:
                print(f"Warning: Hyperparameter search error: {e}")
                print(f"Skipping model {model_name} due to hyperparameter search failure")
                tuner_success = False
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n❌ Error in hyperparameter search for {model_name}: {str(e)}\n")
                
                continue
            
            # Step 10: Cross-validation only if hyperparameter tuning succeeded
            if tuner_success:
                # Simplified cross-validation for memory efficiency
                best_hps = tuner.get_best_hyperparameters(1)
                if best_hps:
                    best_hp = best_hps[0]
                else:
                    print(f"No valid hyperparameters found for {model_name}. Moving to next model.")
                    continue
                
                # Step 11: Train final model with simplified progressive training
                print("\nTraining final model...")
                
                # Build model with best hyperparameters
                final_model = custom_builder(best_hp)
                if final_model is None:
                    print(f"Failed to build final model for {model_name}")
                    continue
                
                # Train model
                history = final_model.fit(
                    train_dataset,
                    epochs=config.max_epochs,
                    steps_per_epoch=train_steps_per_epoch,  # Limit steps per epoch
                    validation_data=test_dataset,
                    validation_steps=val_steps_per_epoch,   # Limit validation steps
                    callbacks=create_callbacks(),
                    class_weight=class_weight_dict,
                    verbose=1
                )
                
                print_memory_usage(f"After training - {model_name}")
                
                # Print best hyperparameters
                print("\nBest Hyperparameters:")
                for param in best_hp.values:
                    print(f"- {param}: {best_hp.values[param]}")
                
                # Step 12: Evaluate the model
                print(f"📊 Evaluating {model_name} model...")
                evaluation = final_model.evaluate(test_dataset, verbose=1)
                
                # Step 13: Save the model
                model_path = f'models/buy_models/{model_name}.h5'
                try:
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    final_model.save(model_path)
                    print(f"Model saved to {model_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                
                # Log results
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n📌 Model: {model_name}\n")
                    log_file.write("=" * 40 + "\n")
                    log_file.write("Hyperparameters:\n")
                    for param in best_hp.values:
                        log_file.write(f"- {param}: {best_hp.values[param]}\n")

                    log_file.write("\nTest Metrics:\n")
                    for i, metric in enumerate(final_model.metrics_names):
                        log_file.write(f"{metric}: {evaluation[i]:.4f}\n")
                    
                    log_file.write("=" * 80 + "\n\n")
        
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            with open(log_file_path, "a") as log_file:
                log_file.write(f"\n❌ Error training {model_name}: {str(e)}\n")
                log_file.write("=" * 80 + "\n")
        
        finally:
            # Clean up after each model to free memory
            tf.keras.backend.clear_session()
            gc.collect()
            print_memory_usage(f"After cleanup - {model_name}")
    
    print(f"\n✅ All models trained! Results saved to {log_file_path}")
    print_memory_usage("End")

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
                        print("GPU detected by nvidia-smi but not by TensorFlow.")
                        print(f"TensorFlow version: {tf.__version__}")
                        print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
                        
                        with tf.device('/CPU:0'):
                            print("\nUsing CPU for computations")
                            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                            print(f"Tensor device: {a.device}")
                
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
    # config.load_config("classification_config.json")
    
    # Start training
    train_classification_models(config)

#%%