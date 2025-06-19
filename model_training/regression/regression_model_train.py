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
import preprocess_data
import shap
import gc
from tqdm import tqdm
import psutil
import time

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ======================================
# OPTIMIZED GPU AND MEMORY CONFIGURATION
# ======================================
def configure_gpu_and_memory():
    """Optimized GPU configuration for A100"""
    print("🔧 Configuring GPU and memory settings...")
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Set memory limit if needed (optional, remove if you want full memory)
            # tf.config.experimental.set_memory_limit(gpus[0], 14000)  # 14GB limit
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Mixed precision for A100 optimization
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("✅ Mixed precision enabled (float16)")
    
    # XLA compilation for performance
    tf.config.optimizer.set_jit(True)
    print("✅ XLA compilation enabled")
    
    # Thread optimization
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    print("✅ Thread optimization configured")

# ======================================
# MEMORY MONITORING UTILITIES
# ======================================
def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # GB

def log_memory_usage(step_name):
    """Log memory usage for debugging"""
    memory_gb = get_memory_usage()
    print(f"💾 Memory after {step_name}: {memory_gb:.2f} GB")
    
    # GPU memory if available
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            gpu_info = tf.config.experimental.get_memory_info('GPU:0')
            gpu_used = gpu_info['current'] / (1024**3)
            print(f"🎮 GPU Memory: {gpu_used:.2f} GB")
    except:
        pass

def cleanup_memory():
    """Force garbage collection and clear TF session"""
    gc.collect()
    tf.keras.backend.clear_session()

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
    """Central configuration manager for regression model training"""
    def __init__(self):
        # Data settings
        self.data_file = 'Close_dataset.csv'
        self.results_file = "regression_training_results.txt"
        
        # Model parameters
        self.lookback_window = 240
        self.forecast_horizon = 15
        self.test_size = 0.2
        self.random_seed = 42
        
        # Training parameters - optimized for A100
        self.batch_size = 512  # Increased for A100
        self.max_epochs = 50
        self.early_stopping_patience = 10
        self.initial_learning_rate = 1e-3

        self.max_trials=150,  # Increase max trials to 150
        self.max_consecutive_failed_trials=30
        
        # Memory optimization settings
        self.chunk_size = 10000  # Process data in chunks
        self.prefetch_buffer = tf.data.AUTOTUNE
        self.num_parallel_calls = tf.data.AUTOTUNE
        
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
# OPTIMIZED DATA LOADING AND PREPROCESSING
# ======================================
def load_and_preprocess_data_optimized(config):
    """Optimized data loading and preprocessing with memory management"""
    print("🔄 Loading and preprocessing data...")
    log_memory_usage("start")
    
    # Load data with optimized dtypes
    print("📂 Loading CSV file...")
    try:
        data = pd.read_csv(config.data_file, header=0)
        print(f"✅ Loaded {len(data):,} rows of data")
        log_memory_usage("data loading")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise
    
    # Basic data cleaning
    print("🧹 Cleaning data...")
    data = preprocess_data.clean_data(data)
    log_memory_usage("data cleaning")
    
    # Optimize data types to save memory
    print("🔧 Optimizing data types...")
    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')
        elif data[col].dtype == 'int64':
            data[col] = data[col].astype('int32')
    log_memory_usage("dtype optimization")
    
    # Process volume features with memory optimization
    print("📊 Processing volume features...")
    volume_cols = ['Volume']
    for col in volume_cols:
        if col in data.columns:
            # Process one transformation at a time to save memory
            print(f"  Processing {col}...")
            
            # Log transform
            data[f'{col}_log'] = np.log1p(data[col].astype(np.float32))
            
            # Winsorize extreme values
            q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
            data[f'{col}_winsor'] = data[col].clip(q_low, q_high).astype(np.float32)
            
            # Rank transform
            data[f'{col}_rank'] = data[col].rank(pct=True).astype(np.float32)
            
            # Clean up intermediate variables
            del q_low, q_high
            gc.collect()
    
    log_memory_usage("volume features")
    
    # Add lag features with memory optimization
    print("⏰ Adding lag features...")
    target_col = 'Close'
    lag_list = config.lag_periods
    
    for lag in tqdm(lag_list, desc="Processing lags"):
        data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag).astype(np.float32)
        
        # Add lagged versions of key technical indicators
        for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
            if indicator in data.columns:
                data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag).astype(np.float32)
    
    # Add rolling stats efficiently
    print("📈 Computing rolling statistics...")
    lag_cols = [f'{target_col}_lag_{lag}' for lag in lag_list]
    data['target_lag_mean'] = data[lag_cols].mean(axis=1).astype(np.float32)
    data['target_lag_std'] = data[lag_cols].std(axis=1).astype(np.float32)
    
    log_memory_usage("lag features")
    
    # Define columns to exclude
    exclude_cols = [
        'Close', 'long_signal', 'short_signal', 'close_position',
        'Date', 'Time', 'datetime'
    ]
    
    # Remove excluded columns that exist in the data
    cols_to_exclude = [col for col in exclude_cols if col in data.columns]
    if cols_to_exclude:
        print(f"🗑️  Excluding columns: {cols_to_exclude}")
        data = data.drop(columns=cols_to_exclude)
    
    # Fill missing values efficiently
    print("🔧 Filling missing values...")
    data = data.bfill().ffill()
    
    # Split features and target
    features = data.copy()
    
    # Reload target separately to save memory
    print("🎯 Loading target variable...")
    original_data = pd.read_csv(config.data_file, header=0, usecols=[target_col])
    original_data = preprocess_data.clean_data(original_data)
    target = original_data[[target_col]].astype(np.float32)
    
    # Ensure same length
    min_length = min(len(features), len(target))
    features = features.iloc[:min_length].copy()
    target = target.iloc[:min_length].copy()
    
    # Force garbage collection
    del data, original_data
    gc.collect()
    
    log_memory_usage("preprocessing complete")
    
    print(f"✅ Final dataset shape: Features: {features.shape}, Target: {target.shape}")
    print(f"📋 Feature columns: {len(features.columns)} features")
    
    return features, target, target_col

# ======================================
# OPTIMIZED TIME SERIES WINDOW CREATION 
# ======================================
def create_sliding_windows_optimized(features, target, n_in, n_out, chunk_size=5000):
    """Memory-efficient sliding window creation with chunking"""
    print(f"🪟 Creating sliding windows: lookback={n_in}, horizon={n_out}")
    log_memory_usage("before windowing")
    
    n_features = features.shape[1]
    n_samples = len(features)
    
    # Calculate final array sizes
    max_samples = n_samples - n_in - n_out + 1
    if max_samples <= 0:
        raise ValueError(f"Not enough data for windowing. Need at least {n_in + n_out} samples, got {n_samples}")
    
    print(f"📊 Will create {max_samples:,} windows from {n_samples:,} samples")
    
    # Convert to float32 arrays
    print("🔄 Converting to optimized arrays...")
    features_values = features.values.astype(np.float32)
    target_values = target.values.astype(np.float32)
    
    log_memory_usage("array conversion")
    
    # Pre-allocate output arrays
    print("📦 Pre-allocating output arrays...")
    features_array = np.empty((max_samples, n_in, n_features), dtype=np.float32)
    target_array = np.empty((max_samples, n_out, 1), dtype=np.float32)
    
    log_memory_usage("array allocation")
    
    # Process in chunks to avoid memory issues
    print(f"⚡ Processing windows in chunks of {chunk_size:,}...")
    
    for start_idx in tqdm(range(0, max_samples, chunk_size), desc="Creating windows"):
        end_idx = min(start_idx + chunk_size, max_samples)
        chunk_length = end_idx - start_idx
        
        # Create feature windows for this chunk
        for i in range(chunk_length):
            sample_idx = start_idx + i
            features_array[sample_idx] = features_values[sample_idx:sample_idx + n_in]
            target_array[sample_idx] = target_values[sample_idx + n_in:sample_idx + n_in + n_out].reshape(-1, 1)
        
        # Periodic memory cleanup
        if start_idx % (chunk_size * 5) == 0:
            gc.collect()
    
    # Handle NaNs efficiently
    print("🔧 Handling missing values...")
    
    # Features NaN handling
    nan_mask = np.isnan(features_array)
    if np.any(nan_mask):
        print(f"⚠️  Found {np.sum(nan_mask):,} NaN values in features")
        # Fill with column means
        features_2d = features_array.reshape(-1, n_features)
        col_means = np.nanmean(features_2d, axis=0)
        
        for i in range(n_features):
            col_nan_mask = nan_mask[:, :, i]
            if np.any(col_nan_mask):
                features_array[col_nan_mask] = col_means[i]
    
    # Target NaN handling
    target_nan_mask = np.isnan(target_array)
    if np.any(target_nan_mask):
        print(f"⚠️  Found {np.sum(target_nan_mask):,} NaN values in targets")
        target_mean = np.nanmean(target_array)
        target_array[target_nan_mask] = target_mean
    
    # Final cleanup
    del features_values, target_values, features, target
    gc.collect()
    
    log_memory_usage("windowing complete")
    print(f"✅ Created windows - Features: {features_array.shape}, Target: {target_array.shape}")
    
    return features_array, target_array

# ======================================
# OPTIMIZED DATA SCALING
# ======================================
def scale_data_optimized(train_X, test_X, train_y, test_y, n_features, chunk_size=5000):
    """Memory-efficient data scaling with chunking"""
    print("⚖️  Scaling features and targets...")
    log_memory_usage("before scaling")
    
    # Scale features in chunks to avoid memory issues
    print("🔄 Scaling features...")
    
    # Fit scaler on training data
    print("  Fitting feature scaler...")
    train_X_sample = train_X[:min(chunk_size, len(train_X))].reshape(-1, n_features)
    train_X_sample = np.nan_to_num(train_X_sample, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler_X = RobustScaler(quantile_range=(10.0, 90.0))
    scaler_X.fit(train_X_sample)
    
    del train_X_sample
    gc.collect()
    
    # Transform training features in chunks
    print("  Transforming training features...")
    for start_idx in tqdm(range(0, len(train_X), chunk_size), desc="Scaling train features"):
        end_idx = min(start_idx + chunk_size, len(train_X))
        
        # Get chunk and reshape
        chunk = train_X[start_idx:end_idx]
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale and clip
        chunk_2d = scaler_X.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)
        
        # Reshape back and assign
        train_X[start_idx:end_idx] = chunk_2d.reshape(chunk.shape)
        
        # Cleanup
        del chunk, chunk_2d
        if start_idx % (chunk_size * 5) == 0:
            gc.collect()
    
    log_memory_usage("train features scaled")
    
    # Transform test features in chunks
    print("  Transforming test features...")
    for start_idx in tqdm(range(0, len(test_X), chunk_size), desc="Scaling test features"):
        end_idx = min(start_idx + chunk_size, len(test_X))
        
        chunk = test_X[start_idx:end_idx]
        chunk_2d = chunk.reshape(-1, n_features)
        chunk_2d = np.nan_to_num(chunk_2d, nan=0.0, posinf=0.0, neginf=0.0)
        
        chunk_2d = scaler_X.transform(chunk_2d)
        chunk_2d = np.clip(chunk_2d, -10, 10)
        
        test_X[start_idx:end_idx] = chunk_2d.reshape(chunk.shape)
        
        del chunk, chunk_2d
        if start_idx % (chunk_size * 5) == 0:
            gc.collect()
    
    log_memory_usage("test features scaled")
    
    # Scale targets
    print("🎯 Scaling targets...")
    scaler_y = RobustScaler(quantile_range=(10.0, 90.0))
    
    # Reshape for scaling
    train_y_2d = train_y.reshape(-1, 1)
    test_y_2d = test_y.reshape(-1, 1)
    
    # Fit and transform
    train_y_2d = scaler_y.fit_transform(train_y_2d)
    test_y_2d = scaler_y.transform(test_y_2d)
    
    # Reshape back
    train_y = train_y_2d.reshape(train_y.shape)
    test_y = test_y_2d.reshape(test_y.shape)
    
    # Cleanup
    del train_y_2d, test_y_2d
    gc.collect()
    
    # Log statistics
    data_stats = {
        'min': float(np.min(train_X)),
        'max': float(np.max(train_X)),
        'mean': float(np.mean(train_X)),
        'std': float(np.std(train_X))
    }
    print(f"📊 Data statistics after scaling: {data_stats}")
    
    log_memory_usage("scaling complete")
    
    return train_X, test_X, train_y, test_y, scaler_X, scaler_y

# ======================================
# OPTIMIZED DATA PIPELINE CREATION
# ======================================
def create_optimized_dataset(X, y, batch_size, shuffle=False, prefetch_buffer=tf.data.AUTOTUNE):
    """Create optimized tf.data.Dataset with proper memory management"""
    print(f"🚀 Creating optimized dataset: batch_size={batch_size}, shuffle={shuffle}")
    
    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Add shuffling if required
    if shuffle:
        # Use a reasonable buffer size to avoid memory issues
        buffer_size = min(10000, len(X))
        dataset = dataset.shuffle(buffer_size, seed=SEED)
    
    # Batch the data
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # Add prefetching for performance
    dataset = dataset.prefetch(prefetch_buffer)
    
    print(f"✅ Dataset created successfully")
    return dataset

# ======================================
# CUSTOM LOSS FUNCTIONS & METRICS (UNCHANGED)
# ======================================
def asymmetric_loss(y_true, y_pred, beta=1.0):
    """Asymmetric loss function that penalizes under-predictions more than over-predictions"""
    error = y_true - y_pred
    under_forecast = tf.maximum(tf.zeros_like(error), error)
    over_forecast = tf.maximum(tf.zeros_like(error), -error)
    
    loss = tf.reduce_mean(beta * tf.square(under_forecast) + tf.square(over_forecast))
    return loss

def directional_accuracy_metric(y_true, y_pred):
    """Measures percentage of times the prediction direction matches actual direction"""
    true_direction = tf.sign(y_true[:, 1:] - y_true[:, :-1])
    pred_direction = tf.sign(y_pred[:, 1:] - y_pred[:, :-1])
    
    correct_direction = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)
    accuracy = tf.reduce_mean(correct_direction)
    return accuracy

# ======================================
# LEARNING RATE SCHEDULERS AND CALLBACKS
# ======================================
def cosine_annealing_warmup_schedule(epoch, lr, total_epochs=50, warmup_epochs=5, min_lr=1e-6):
    """Cosine annealing with warmup learning rate schedule"""
    if epoch < warmup_epochs:
        return lr * ((epoch + 1) / warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

class OptimizedGPUMemoryCallback(tf.keras.callbacks.Callback):
    """Optimized GPU memory monitoring callback"""
    def on_epoch_end(self, epoch, logs=None):
        try:
            log_memory_usage(f"epoch {epoch}")
        except Exception as e:
            pass  # Silent fail to avoid disrupting training

def get_optimized_training_callbacks():
    """Create optimized callbacks for model training"""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: cosine_annealing_warmup_schedule(epoch, lr),
            verbose=0
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        OptimizedGPUMemoryCallback()
    ]
    return callbacks

# ======================================
# OPTIMIZED MODEL TRAINING UTILITIES
# ======================================
def implement_progressive_training_optimized(model_builder, hp, train_dataset, test_dataset, 
                                           callbacks, model_name, steps_per_epoch):
    """Optimized progressive training with proper memory management"""
    print(f"\n⚙️ Implementing optimized progressive training for {model_name}...")
    log_memory_usage("training start")
    
    # Build model with error handling
    try:
        model = model_builder(hp)
        if model is None:
            print(f"❌ Model builder returned None for {model_name}")
            return None, None
    except Exception as e:
        print(f"❌ Error building model for {model_name}: {e}")
        return None, None
    
    # Phase 1: Initial training on subset
    subset_steps = max(1, steps_per_epoch // 5)
    print(f"🔄 Phase 1: Training on {subset_steps} steps ({subset_steps/steps_per_epoch:.1%} of data)")
    
    try:
        # Take subset of training data
        train_subset = train_dataset.take(subset_steps)
        
        initial_history = model.fit(
            train_subset,
            epochs=15,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=subset_steps
        )
        
        print(f"✅ Phase 1 completed")
        log_memory_usage("phase 1 complete")
        
    except Exception as e:
        print(f"❌ Error in Phase 1 training for {model_name}: {e}")
        return None, None
    
    # Phase 2: Full dataset fine-tuning
    try:
        print(f"\n🔄 Phase 2: Fine-tuning on full dataset")
        
        # Reduce learning rate for fine-tuning
        try:
            if hasattr(model.optimizer, 'learning_rate'):
                current_lr = model.optimizer.learning_rate
                if hasattr(current_lr, 'numpy'):
                    current_lr = float(current_lr.numpy())
                elif isinstance(current_lr, (int, float)):
                    current_lr = float(current_lr)
                else:
                    current_lr = 0.0001
                
                # Recompile with reduced learning rate
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr * 0.5),
                    loss=model.loss,
                    metrics=model.metrics
                )
                print(f"🔧 Reduced learning rate to {current_lr * 0.5:.6f}")
        
        except Exception as e:
            print(f"⚠️  Error adjusting learning rate: {e}")
        
        # Full dataset training
        final_history = model.fit(
            train_dataset,
            epochs=35,
            validation_data=test_dataset,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch
        )
        
        print(f"✅ Phase 2 completed")
        log_memory_usage("training complete")
        
        return model, final_history
    
    except Exception as e:
        print(f"❌ Error in Phase 2 training for {model_name}: {e}")
        return None, None

def get_custom_model_builder_optimized(model_name, model_builder, n_out, input_shape):
    """Optimized custom model builder with better error handling"""
    def custom_model_builder(hp):
        try:
            print(f"🏗️  Building {model_name} model with input shape: {input_shape}")
            
            # Get builder function
            if isinstance(model_builder, str):
                if model_builder in MODEL_BUILDERS:
                    builder_func = MODEL_BUILDERS[model_builder]
                else:
                    print(f"❌ Unknown model type: {model_builder}")
                    return None
            else:
                builder_func = model_builder
            
            # Try different parameter combinations
            try:
                # Create a dummy train_X for compatibility
                dummy_train_X = np.zeros((1, *input_shape[1:]), dtype=np.float32)
                base_model = builder_func(hp, train_X=dummy_train_X, n_out=n_out)
            except TypeError:
                try:
                    base_model = builder_func(hp, dummy_train_X, n_out)
                except TypeError:
                    try:
                        base_model = builder_func(hp)
                    except Exception as e:
                        print(f"❌ Error creating {model_name} model: {e}")
                        return None
            
            if base_model is None:
                print(f"❌ Model builder for {model_name} returned None")
                return None

            # Add regularization for stability
            for layer in base_model.layers:
                if hasattr(layer, 'kernel_regularizer') and isinstance(layer, tf.keras.layers.Dense):
                    layer.kernel_regularizer = tf.keras.regularizers.l2(1e-5)

            # Configure loss function
            loss_type = hp.Choice("loss_type", ["huber", "mse", "asymmetric"], default="mse")
            
            if loss_type == "huber":
                loss_fn = tf.keras.losses.Huber()
            elif loss_type == "asymmetric":
                beta = hp.Float("asymmetric_beta", min_value=1.0, max_value=3.0, step=0.5, default=1.5)
                loss_fn = lambda y_true, y_pred: asymmetric_loss(y_true, y_pred, beta=beta)
            else:
                loss_fn = 'mse'
            
            # Optimized learning rate for A100
            learning_rate = hp.Float("learning_rate", min_value=1e-5, max_value=5e-3, 
                                    sampling="log", default=1e-3)
            
            # Compile model with optimizations
            base_model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    clipnorm=1.0,
                    epsilon=1e-7  # Better numerical stability
                ),
                loss=loss_fn,
                metrics=['mse', 'mae'],
                jit_compile=True
            )
            
            return base_model
                
        except Exception as e:
            print(f"❌ Error in custom model builder for {model_name}: {e}")
            return None

    return custom_model_builder

# ======================================
# OPTIMIZED CROSS VALIDATION
# ======================================
def cross_validate_best_models_optimized(tuner, train_X, train_y, test_X, test_y, model_name, n_folds=3):
    """Optimized cross-validation with reduced folds for speed"""
    print(f"🔄 Running {n_folds}-fold cross-validation on top models...")
    log_memory_usage("CV start")
    
    try:
        top_hps = tuner.get_best_hyperparameters(2)  # Reduced from 3 to 2
        if not top_hps:
            print("❌ No valid hyperparameters found from tuning")
            return None, None
    except Exception as e:
        print(f"❌ Error getting hyperparameters: {e}")
        return None, None
    
    kf = KFold(n_splits=n_folds, shuffle=False, random_state=42)
    cv_results = []
    
    for hp_idx, hp in enumerate(top_hps):
        print(f"\n🧪 Validating hyperparameter set {hp_idx+1}/{len(top_hps)}")
        
        fold_metrics = {'val_loss': [], 'val_mse': [], 'val_mae': []}
        
        custom_builder = get_custom_model_builder_optimized(
            model_name, MODEL_BUILDERS[model_name], N_OUT, train_X.shape
        )
        
        all_folds_successful = True
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_X)):
            print(f"  📁 Fold {fold+1}/{n_folds}")
            
            try:
                # Split data efficiently
                X_train_fold, X_val_fold = train_X[train_idx], train_X[val_idx]
                y_train_fold, y_val_fold = train_y[train_idx], train_y[val_idx]
                
                # Create optimized datasets
                train_fold_dataset = create_optimized_dataset(X_train_fold, y_train_fold, 256, shuffle=False)
                val_fold_dataset = create_optimized_dataset(X_val_fold, y_val_fold, 256, shuffle=False)
                
                # Build model
                model = custom_builder(hp)
                if model is None:
                    print(f"  ❌ Model creation failed for fold {fold+1}")
                    all_folds_successful = False
                    break
                
                callbacks = get_optimized_training_callbacks()
                
                # Calculate steps
                steps_per_epoch = max(1, len(X_train_fold) // 256)
                
                # Train model with reduced epochs for CV
                history = model.fit(
                    train_fold_dataset,
                    epochs=8,  # Reduced from 10
                    validation_data=val_fold_dataset,
                    callbacks=callbacks,
                    verbose=0,
                    steps_per_epoch=steps_per_epoch
                )
                
                # Collect metrics
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
                    all_folds_successful = False
                    break
                    
            except Exception as e:
                print(f"  ❌ Error in fold {fold+1}: {str(e)}")
                all_folds_successful = False
                break
            finally:
                cleanup_memory()
        
        if not all_folds_successful or not fold_metrics['val_loss']:
            print(f"  ❌ Hyperparameter set {hp_idx+1} failed validation")
            continue
            
        # Calculate statistics
        cv_summary = {}
        for metric, values in fold_metrics.items():
            if all(np.isnan(values)):
                cv_summary[f'{metric}_mean'] = np.nan
                cv_summary[f'{metric}_std'] = np.nan
            else:
                filtered_values = [v for v in values if not np.isnan(v)]
                if filtered_values:
                    cv_summary[f'{metric}_mean'] = np.mean(filtered_values)
                    cv_summary[f'{metric}_std'] = np.std(filtered_values) if len(filtered_values) > 1 else 0.0
                else:
                    cv_summary[f'{metric}_mean'] = np.nan
                    cv_summary[f'{metric}_std'] = np.nan
        
        cv_summary['hyperparameters'] = hp
        cv_results.append(cv_summary)
        
        print(f"  ✅ Results: val_loss_mean={cv_summary.get('val_loss_mean', np.nan):.4f} ± {cv_summary.get('val_loss_std', np.nan):.4f}")
    
    # Select best hyperparameters
    if cv_results:
        valid_results = [r for r in cv_results if not np.isnan(r.get('val_loss_mean', np.nan))]
        
        if valid_results:
            valid_results.sort(key=lambda x: x.get('val_loss_mean', float('inf')))
            best_cv_result = valid_results[0]
            best_hp = best_cv_result['hyperparameters']
            
            print(f"\n✅ Best cross-validated model:")
            print(f"  val_loss: {best_cv_result.get('val_loss_mean', np.nan):.4f} ± {best_cv_result.get('val_loss_std', np.nan):.4f}")
            
            return best_hp, valid_results
    
    print("❌ No valid cross-validation results")
    return None, None

# ======================================
# MODEL EVALUATION UTILITIES (OPTIMIZED)
# ======================================
def evaluate_forecasts_optimized(y_pred, test_y, scaler_y, n_out):
    """Optimized forecast evaluation with memory management"""
    print("📊 Evaluating forecasts...")
    log_memory_usage("evaluation start")
    
    # Inverse transform with memory optimization
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

    # Calculate metrics efficiently
    rmse_values = []
    mae_values = []
    r2_values = []
    
    print(f"📐 Shapes - True: {test_y_inv.shape}, Pred: {y_pred_inv.shape}")
    
    # Handle different prediction shapes efficiently
    if len(test_y_inv.shape) == 3 and len(y_pred_inv.shape) == 2 and y_pred_inv.shape[1] == 1:
        print("📈 Computing metrics for single-step prediction")
        
        for i in range(min(n_out, test_y_inv.shape[1])):
            step_true = test_y_inv[:, i, 0]
            step_pred = y_pred_inv[:, 0]
            
            step_rmse = sqrt(mean_squared_error(step_true, step_pred))
            step_mae = mean_absolute_error(step_true, step_pred)
            step_r2 = r2_score(step_true, step_pred)
            
            rmse_values.append(step_rmse)
            mae_values.append(step_mae)
            r2_values.append(step_r2)
        
        overall_rmse = np.mean(rmse_values)
        overall_mae = np.mean(mae_values)
        overall_r2 = np.mean(r2_values)
        
    else:
        print("📈 Computing metrics for multi-step prediction")
        
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
        
        # Compute overall metrics
        try:
            test_y_flat = test_y_inv.flatten()
            y_pred_flat = y_pred_inv.flatten()
            
            if len(test_y_flat) == len(y_pred_flat):
                overall_rmse = sqrt(mean_squared_error(test_y_flat, y_pred_flat))
                overall_mae = mean_absolute_error(test_y_flat, y_pred_flat)
                overall_r2 = r2_score(test_y_flat, y_pred_flat)
            else:
                overall_rmse = np.mean(rmse_values)
                overall_mae = np.mean(mae_values)
                overall_r2 = np.mean(r2_values)
        except:
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
    
    log_memory_usage("evaluation complete")
    return evaluation_results

# ======================================
# MAIN OPTIMIZED TRAINING PIPELINE
# ======================================
def train_regression_models_optimized(config):
    """Optimized main training pipeline with comprehensive memory management"""
    print("🚀 Starting optimized regression model training...")
    
    # Configure GPU and memory
    configure_gpu_and_memory()
    log_memory_usage("initialization")
    
    # Set up logging
    log_file_path = config.results_file
    with open(log_file_path, "w") as log_file:
        log_file.write("Optimized Multi-step Regression Models Training Results\n")
        log_file.write("=" * 80 + "\n\n")
    
    try:
        # Step 1: Load and preprocess data
        print("\n" + "="*50)
        print("📂 STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*50)
        
        features, target, target_col = load_and_preprocess_data_optimized(config)
        
        # Step 2: Create sliding windows
        print("\n" + "="*50)
        print("🪟 STEP 2: CREATING SLIDING WINDOWS")
        print("="*50)
        
        features_array, target_array = create_sliding_windows_optimized(
            features, target, config.lookback_window, config.forecast_horizon, 
            chunk_size=config.chunk_size
        )
        
        # Clean up original data
        del features, target
        gc.collect()
        log_memory_usage("after windowing cleanup")
        
        # Step 3: Split into train/test sets
        print("\n" + "="*50)
        print("✂️  STEP 3: TRAIN/TEST SPLIT")
        print("="*50)
        
        train_X, test_X, train_y, test_y = train_test_split(
            features_array, target_array, 
            test_size=config.test_size, 
            random_state=config.random_seed, 
            shuffle=False
        )
        
        # Clean up original arrays
        del features_array, target_array
        gc.collect()
        log_memory_usage("after split cleanup")
        
        print(f"📊 Split shapes - Train: {train_X.shape}, Test: {test_X.shape}")
        
        # Step 4: Scale the data
        print("\n" + "="*50)
        print("⚖️  STEP 4: DATA SCALING")
        print("="*50)
        
        n_features = train_X.shape[2]
        train_X, test_X, train_y, test_y, scaler_X, scaler_y = scale_data_optimized(
            train_X, test_X, train_y, test_y, n_features, chunk_size=config.chunk_size
        )
        
        # Step 5: Create optimized datasets
        print("\n" + "="*50)
        print("🚀 STEP 5: CREATING OPTIMIZED DATASETS")
        print("="*50)
        
        train_dataset = create_optimized_dataset(
            train_X, train_y, config.batch_size, shuffle=False, prefetch_buffer=config.prefetch_buffer
        )
        test_dataset = create_optimized_dataset(
            test_X, test_y, config.batch_size, shuffle=False, prefetch_buffer=config.prefetch_buffer
        )
        
        # Calculate steps per epoch
        steps_per_epoch = max(1, len(train_X) // config.batch_size)
        print(f"📊 Steps per epoch: {steps_per_epoch}")
        
        log_memory_usage("datasets created")
        
        # Get list of models to train
        if config.selected_models:
            model_list = config.selected_models
        else:
            model_list = list(MODEL_BUILDERS.keys())
        
        print(f"\n🎯 Will train {len(model_list)} models: {model_list}")
        
        # Step 6: Train models
        print("\n" + "="*50)
        print("🏋️  STEP 6: MODEL TRAINING")
        print("="*50)
        
        successful_models = 0
        failed_models = 0
        
        for model_idx, model_name in enumerate(model_list):
            print(f"\n\n{'='*60}")
            print(f"🚀 Training model {model_idx+1}/{len(model_list)}: {model_name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            try:
                # Log model training start
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n\n## Model: {model_name}\n")
                    log_file.write("=" * 50 + "\n")
                
                log_memory_usage(f"before {model_name}")
                
                # Get model builder and callbacks
                model_builder = MODEL_BUILDERS[model_name]
                callbacks = get_optimized_training_callbacks()
                custom_builder = get_custom_model_builder_optimized(
                    model_name, model_builder, config.forecast_horizon, train_X.shape
                )
                
                # Step 6a: Hyperparameter tuning
                print(f"🔍 Hyperparameter search for {model_name}...")
                tuner = kt.Hyperband(
                    hypermodel=lambda hp: custom_builder(hp),
                    objective='val_loss',
                    max_epochs=20,  # You can keep this or adjust as needed
                    factor=3,
                    hyperband_iterations=1,
                    directory='models_dir',
                    project_name=f'{model_name}_regression_tuning',
                    overwrite=True,
                    max_consecutive_failed_trials=30  # Set max consecutive failed trials to 30
                )
                
                tuner_success = True
                try:
                    tuner.search(
                        train_dataset,
                        epochs=8,  # Reduced from 10
                        validation_data=test_dataset,
                        callbacks=callbacks,
                        verbose=1
                    )
                    print(f"✅ Hyperparameter search completed for {model_name}")
                    
                except Exception as e:
                    print(f"❌ Hyperparameter search error for {model_name}: {e}")
                    tuner_success = False
                    
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"\n❌ Error in hyperparameter search: {str(e)}\n")
                    
                    failed_models += 1
                    continue
                
                # Step 6b: Cross-validation
                if tuner_success:
                    print(f"🧪 Cross-validation for {model_name}...")
                    best_hp, cv_results = cross_validate_best_models_optimized(
                        tuner, train_X, train_y, test_X, test_y, model_name, n_folds=3
                    )
                    
                    if best_hp is None:
                        print(f"❌ Cross-validation failed for {model_name}")
                        
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"\n❌ Cross-validation failed\n")
                        
                        failed_models += 1
                        continue
                    
                    print(f"✅ Cross-validation completed for {model_name}")
                    
                    # Step 6c: Train final model
                    print(f"🏋️  Final training for {model_name}...")
                    best_model, history = implement_progressive_training_optimized(
                        model_builder=lambda hp: custom_builder(best_hp),
                        hp=best_hp,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset,
                        callbacks=callbacks,
                        model_name=model_name,
                        steps_per_epoch=steps_per_epoch
                    )
                    
                    if best_model is None or history is None:
                        print(f"❌ Final training failed for {model_name}")
                        
                        with open(log_file_path, "a") as log_file:
                            log_file.write(f"\n❌ Final training failed\n")
                        
                        failed_models += 1
                        continue
                    
                    print(f"✅ Final training completed for {model_name}")
                    
                    # Step 6d: Generate predictions and evaluate
                    print(f"📊 Evaluating {model_name}...")
                    y_pred = best_model.predict(test_X, batch_size=config.batch_size, verbose=0)
                    evaluation_results = evaluate_forecasts_optimized(
                        y_pred, test_y, scaler_y, config.forecast_horizon
                    )
                    
                    print(f"📈 {model_name} Performance:")
                    print(f"  RMSE: {evaluation_results['overall_rmse']:.4f}")
                    print(f"  MAE: {evaluation_results['overall_mae']:.4f}")
                    print(f"  R²: {evaluation_results['overall_r2']:.4f}")
                    
                    # Step 6e: Save the model
                    os.makedirs('models', exist_ok=True)
                    model_path = f'models/{model_name}_regressor.keras'
                    best_model.save(model_path)
                    print(f"💾 Model saved to {model_path}")
                    
                    # Log results
                    training_time = time.time() - start_time
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"\n📌 Model: {model_name}\n")
                        log_file.write("=" * 40 + "\n")
                        log_file.write(f"Training time: {training_time:.2f} seconds\n")
                        
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
                    
                    successful_models += 1
                    print(f"✅ {model_name} completed successfully in {training_time:.2f}s")
                    
                else:
                    failed_models += 1
            
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                import traceback
                traceback.print_exc()
                
                with open(log_file_path, "a") as log_file:
                    log_file.write(f"\n❌ Error training {model_name}: {str(e)}\n")
                    log_file.write("=" * 80 + "\n")
                
                failed_models += 1
            
            finally:
                # Clean up memory after each model
                cleanup_memory()
                log_memory_usage(f"after {model_name} cleanup")
        
        # Final summary
        print(f"\n🎉 Training completed!")
        print(f"✅ Successful models: {successful_models}")
        print(f"❌ Failed models: {failed_models}")
        print(f"📊 Success rate: {successful_models/(successful_models+failed_models)*100:.1f}%")
        print(f"📄 Results saved to {log_file_path}")
        
        log_memory_usage("final")
    
    except Exception as e:
        print(f"❌ Fatal error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        with open(log_file_path, "a") as log_file:
            log_file.write(f"\n❌ Fatal error: {str(e)}\n")
        
        raise

# ======================================
# MAIN EXECUTION
# ======================================
if __name__ == "__main__":
    print("🚀 Starting Optimized Regression Model Training")
    print("=" * 60)
    
    # Initialize and start training
    train_regression_models_optimized(config)