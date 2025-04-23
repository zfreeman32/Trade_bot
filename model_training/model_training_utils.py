import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from numpy.lib.stride_tricks import sliding_window_view

# Data loading and preprocessing
def clean_and_enhance_features(data, target_col, lag_list, important_features_file=None):
    """
    Clean data and add enhanced features like log transforms and lags
    
    Args:
        data: DataFrame with raw data
        target_col: Name of target column
        lag_list: List of lag periods to create
        important_features_file: Optional path to file with important features
        
    Returns:
        Processed DataFrame, list of features, target variable
    """
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
    for lag in lag_list:
        data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
        
        # Also add lagged versions of key technical indicators
        for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
            if indicator in data.columns:
                data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
    
    # Add rolling stats on lagged features
    data['target_lag_mean'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].mean(axis=1)
    data['target_lag_std'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].std(axis=1)
    
    # Feature selection - load important features from file if provided
    important_features = []
    if important_features_file:
        try:
            with open(important_features_file, 'r') as f:
                important_features = [line.strip() for line in f.readlines()]
            print(f"Loaded {len(important_features)} important features from file")
        except FileNotFoundError:
            print(f"Important features file not found. Using all features.")
    
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
        # For classification, we need to ensure target cols are kept
        if "short_signal" in data.columns or "long_signal" in data.columns:
            columns_to_keep = [col for col in important_features if col in data.columns]
            columns_to_keep.extend(['long_signal', 'short_signal', 'close_position'])
            columns_to_keep = list(set(columns_to_keep))  # Remove duplicates
            filtered_data = data[columns_to_keep]
        else:
            # For regression, we just keep important features
            filtered_data = data[[col for col in important_features if col in data.columns]]
    else:
        filtered_data = data
    
    # Fill missing values
    filtered_data = filtered_data.fillna(method='bfill').fillna(method='ffill')
    
    return filtered_data

def create_sliding_windows(features, target, n_in, n_out=1, is_regression=True):
    """
    Create sliding windows for time series modeling
    
    Args:
        features: DataFrame with features
        target: Series or DataFrame with target variable(s)
        n_in: Size of lookback window
        n_out: Number of future steps to predict (for regression)
        is_regression: Whether this is a regression task (vs. classification)
        
    Returns:
        Windowed features and targets as numpy arrays
    """
    print(f"Creating sliding windows with lookback={n_in}" + 
          (f", horizon={n_out}" if is_regression else ""))
    
    n_features = features.shape[1]
    features_values = features.values.astype(np.float32)
    
    # Create sliding windows for features
    features_array = sliding_window_view(features_values, n_in, axis=0)
    
    if is_regression:
        # For regression: create sequence of future values to predict
        target_windows = []
        for i in range(len(features) - n_in - n_out + 1):
            target_window = target.values[i + n_in:i + n_in + n_out].astype(np.float32)
            target_windows.append(target_window)
        
        target_array = np.array(target_windows)
    else:
        # For classification: predict single point after window
        target_array = target.values[n_in-1:]
    
    # Ensure features_array matches target_array in length
    features_array = features_array[:len(target_array)]
    
    # Handle missing values and infinities
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
    
    if is_regression:
        target_array = np.where(~np.isfinite(target_array), np.nan, target_array)
        if np.isnan(target_array).sum() > 0:
            print(f"Replacing {np.isnan(target_array).sum()} NaN values in target.")
            target_array = np.nan_to_num(target_array, nan=np.nanmean(target_array))
    
    print(f"Feature windows shape: {features_array.shape}, Target shape: {target_array.shape}")
    return features_array, target_array

def scale_data(train_X, test_X, train_y, test_y, n_features, is_regression=True):
    """
    Apply RobustScaler to features and targets
    
    Args:
        train_X, test_X: Training and test features
        train_y, test_y: Training and test targets
        n_features: Number of features
        is_regression: Whether this is a regression task
        
    Returns:
        Scaled data and scalers
    """
    print("Scaling features" + (" and targets..." if is_regression else "..."))
    
    # Reshape to 2D for scaling
    train_X_2d = train_X.reshape(-1, n_features)
    train_X_2d = np.nan_to_num(train_X_2d, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features with RobustScaler
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
    
    # For regression, also scale targets
    if is_regression:
        scaler_y = RobustScaler(quantile_range=(10.0, 90.0))
        train_y_2d = train_y.reshape(-1, 1) if len(train_y.shape) == 1 else train_y.reshape(-1, train_y.shape[-1])
        test_y_2d = test_y.reshape(-1, 1) if len(test_y.shape) == 1 else test_y.reshape(-1, test_y.shape[-1])
        
        train_y_2d = scaler_y.fit_transform(train_y_2d)
        test_y_2d = scaler_y.transform(test_y_2d)
        
        # Reshape back to original shape
        train_y = train_y_2d.reshape(train_y.shape)
        test_y = test_y_2d.reshape(test_y.shape)
        
        return train_X, test_X, train_y, test_y, scaler_X, scaler_y
    else:
        # For classification, no need to scale targets
        return train_X, test_X, train_y, test_y, scaler_X, None

# Custom loss functions
def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss implementation for classification
    Helps with class imbalance by focusing more on hard examples
    
    Args:
        gamma: Focusing parameter - higher gamma gives more weight to hard examples
        alpha: Class weight parameter for positive class
        
    Returns:
        Loss function compatible with TensorFlow/Keras
    """
    def focal_loss_fn(y_true, y_pred):
        epsilon = 1e-7  # Prevent log(0) errors
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)

        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        loss = alpha_factor * modulating_factor * cross_entropy
        
        # Add small constant to prevent numerical issues
        loss = loss + 1e-8
        
        # Replace NaNs with zero loss
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)

        return tf.reduce_mean(loss)

    return focal_loss_fn

def asymmetric_loss(y_true, y_pred, beta=1.0):
    """
    Asymmetric loss function for regression
    Penalizes under-predictions more than over-predictions (or vice versa)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        beta: Asymmetry parameter (>1 penalizes under-prediction more)
        
    Returns:
        Loss value
    """
    error = y_true - y_pred
    under_forecast = tf.maximum(tf.zeros_like(error), error)
    over_forecast = tf.maximum(tf.zeros_like(error), -error)
    
    loss = tf.reduce_mean(beta * tf.square(under_forecast) + tf.square(over_forecast))
    return loss

# Training callbacks
def get_training_callbacks(is_regression=True):
    """
    Create standard callbacks for model training
    
    Args:
        is_regression: Whether this is a regression task
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5 if is_regression else 3,
            restore_best_weights=True,
            min_delta=0.001
        ),
        
        # Learning rate reduction on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN()
    ]
    
    if is_regression:
        # Add cosine annealing scheduler for regression
        def cosine_annealing(epoch, lr, total_epochs=50, warmup_epochs=5, min_lr=1e-6):
            if epoch < warmup_epochs:
                return lr * ((epoch + 1) / warmup_epochs)
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: cosine_annealing(epoch, lr)
            )
        )
    else:
        # Add simpler learning rate schedule for classification
        def lr_schedule(epoch, lr):
            if epoch < 3:  # Warm-up phase
                return lr * 1.1
            return lr
            
        callbacks.append(
            tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        )
    
    return callbacks

# Progressive training implementation
def implement_progressive_training(model_builder, hp, train_X, train_y, test_X, test_y, 
                                  callbacks, model_name, class_weight_dict=None):
    """
    Implement progressive training - train first on subset, then full dataset
    
    Args:
        model_builder: Function that builds the model given hyperparameters
        hp: Hyperparameters
        train_X, train_y: Training data
        test_X, test_y: Validation data
        callbacks: List of callbacks
        model_name: Name of model for logging
        class_weight_dict: Optional class weights for classification
        
    Returns:
        Trained model and training history
    """
    print(f"\n⚙️ Implementing progressive training for {model_name}...")
    
    # Phase 1: Initial training on smaller subset
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
        callbacks=callbacks,
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
    try:
        K = tf.keras.backend
        current_lr = K.get_value(model.optimizer.learning_rate)
        K.set_value(model.optimizer.learning_rate, current_lr * 0.5)
        print(f"Reducing learning rate from {current_lr:.6f} to {current_lr * 0.5:.6f} for fine-tuning")
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