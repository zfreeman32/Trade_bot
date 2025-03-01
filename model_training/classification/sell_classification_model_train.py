# %%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import keras_tuner as kt
import tensorflow as tf
import shap
from sklearn.utils.class_weight import compute_class_weight
from numpy.lib.stride_tricks import sliding_window_view
from imblearn.over_sampling import SMOTE
from data import preprocess_data
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
    build_TCN_classifier
)

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

MODEL_TYPE = "LSTM"  # Options: LSTM, GRU, Conv1D, Conv1D_LSTM, BiLSTM_Attention, Transformer, MultiStream, ResNet, TCN

# Dictionary of model builders
MODEL_BUILDERS = {
    "LSTM": build_LSTM_classifier,
    "GRU": build_GRU_classifier,
    "Conv1D": build_Conv1D_classifier,
    "Conv1D_LSTM": build_Conv1D_LSTM_classifier,
    "BiLSTM_Attention": build_BiLSTM_Attention_classifier,
    "Transformer": build_Transformer_classifier,
    "MultiStream": build_MultiStream_classifier,
    "ResNet": build_ResNet_classifier,
    "TCN": build_TCN_classifier
}

# %%
# Classification training
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_features.csv'
print(f"Loading data from {file_path}...")
data = pd.read_csv(file_path, header=0)

# Optional: Use subset for faster testing
data = data.tail(1000)
print(f"Using {len(data)} rows of data")

# Preprocess the data
print("Preprocessing data...")
data = preprocess_data.clean_data(data)

#----------------------- ADD THESE SECTIONS -----------------------#

# 1. HANDLE VOLUME-BASED FEATURES WITH HIGH OUTLIERS
print("Transforming volume-based features...")
volume_cols = ['Volume', 'vol_av', 'vol_last_av', 'vol_ratio_av', 'rolling_volume', 'norm_vol']

for col in volume_cols:
    if col in data.columns:
        # Log transform (handles high skewness)
        data[f'{col}_log'] = np.log1p(data[col])
        
        # Winsorize extreme values (cap at percentiles)
        q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
        data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
        
        # Rank transform (completely resistant to outliers)
        data[f'{col}_rank'] = data[col].rank(pct=True)

# 2. ADD LAG FEATURES BASED ON PERIODIC PATTERNS
print("Adding lag features...")
target_col = 'short_signal'
lag_list = [70, 24, 29, 34, 39]  # Based on your analysis

for lag in lag_list:
    data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
    
    # Also add lagged versions of key technical indicators
    for indicator in ['RSI', 'CCI', 'EFI']:
        if indicator in data.columns:
            data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)

# Add rolling stats on lagged features
data['target_lag_mean'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].mean(axis=1)
data['target_lag_std'] = data[[f'{target_col}_lag_{lag}' for lag in lag_list]].std(axis=1)

# 3. FEATURE SELECTION BASED ON IMPORTANCE
print("Performing feature selection...")
# Load pre-computed feature importance if available, otherwise use basic selection
# This is based on your analysis results.txt
important_features = [
    'bb_short_entry_signal', 'stiffness_strat_sell_signal', 'moving_average_buy_signal',
    'Bullish', 'EFI', 'CCI', 'RSI', 'ROC', 'Offset', 'PLUS_DM', 'WILLR', 'vol_ratio_av',
    'price_range', 'ATR', 'MOM', 'Volume', 'Volume_log', 'Volume_winsor', 'Volume_rank'
]
# Add the lag features we created
important_features.extend([f'{target_col}_lag_{lag}' for lag in lag_list])
important_features.extend(['target_lag_mean', 'target_lag_std'])

# Keep only important features + any new transformed features
all_cols = list(data.columns)
# Add any transformed/new features we created above
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

#----------------------- END OF ADDITIONS -----------------------#

# Split features and target
features = data.drop(columns=['long_signal', 'short_signal', 'close_position'])  # All features except targets
target = data[['short_signal']]  # Target is 'short_signal'

# Ensure target is categorical (0 or 1)
target = target.astype(int)

# Compute **class weights** dynamically
classes = np.array([0, 1])  # No-signal vs. short-signal
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=target.values.flatten()
)

class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
print(f"Computed class weights: {class_weight_dict}")

# Configuration for window size
n_in = 240  # Number of past observations (lookback window)
n_features = features.shape[1]  # Number of feature columns

print(f"Features shape after enhancement: {features.shape}, Target shape: {target.shape}")
print(f"Using lookback window of {n_in} timesteps")

# Apply sliding window to create time-series data
print("Creating sliding windows...")
features_array = sliding_window_view(features.values, n_in, axis=0)

# Match target with end of window
target_array = target.values[n_in-1:]  # Target at the end of each window
features_array = features_array[:len(target_array)]  # Trim features to match target length

# Handle NaN and Inf values
features_array = np.where(np.isinf(features_array), np.nan, features_array)
if np.isnan(features_array).sum() > 0:
    print(f"Warning: Found {np.isnan(features_array).sum()} NaN values. Replacing with column means.")
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

print(f"Window data shape: {features_array.shape}")
print(f"Target data shape: {target_array.shape}")

# Train-test split (maintain time sequence)
print("Splitting data into train and test sets...")
train_X, test_X, train_y, test_y = train_test_split(
    features_array, target_array, test_size=0.2, random_state=seed, shuffle=False
)

print("Scaling features...")

# Reshape to 2D for scaling
train_X_2d = train_X.reshape(-1, n_features)

# Check for infinity or NaN values before scaling
if np.isinf(train_X_2d).any() or np.isnan(train_X_2d).any():
    print("Warning: Found Inf or NaN values in training data before scaling. Replacing...")
    train_X_2d = np.nan_to_num(train_X_2d, nan=0.0, posinf=0.0, neginf=0.0)

# Use RobustScaler and fit on training data only
scaler = RobustScaler(quantile_range=(5.0, 95.0))  # More robust to outliers
train_X_2d = scaler.fit_transform(train_X_2d)

# Cap extreme values after scaling
train_X_2d = np.clip(train_X_2d, -10, 10)

# Reshape back to 3D
train_X = train_X_2d.reshape(train_X.shape)

# Apply same scaling to test data
test_X_2d = test_X.reshape(-1, n_features)

# Replace any Inf/NaN values in test data
if np.isinf(test_X_2d).any() or np.isnan(test_X_2d).any():
    print("Warning: Found Inf or NaN values in test data before scaling. Replacing...")
    test_X_2d = np.nan_to_num(test_X_2d, nan=0.0, posinf=0.0, neginf=0.0)

# Transform test data using fitted scaler
test_X_2d = scaler.transform(test_X_2d)

# Cap extreme values in test data too
test_X_2d = np.clip(test_X_2d, -10, 10)

# Reshape back to 3D
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

#----------------------- ADD FOCAL LOSS FOR NEURAL NETWORKS -----------------------#
# Define a focal loss function for neural network training
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        # Clip prediction values to avoid log(0) error - use smaller epsilon
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Convert data types to float32 to ensure numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Binary cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Focal scaling factor with safe calculations
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1.0), alpha, 1.0 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Calculate focal loss
        loss = alpha_factor * modulating_factor * cross_entropy
        
        # Add a small constant to prevent complete zeros
        loss = loss + 1e-8
        
        # Return mean of loss to reduce to a scalar
        return tf.reduce_mean(loss)
    
    return focal_loss_fn

nan_callback = tf.keras.callbacks.TerminateOnNaN()
#----------------------- END OF ADDITIONS -----------------------#

# %%
# Set up the model tuner
print(f"Setting up {MODEL_TYPE} model for hyperparameter tuning...")

try:
    # Create the tuner using the selected model
    model_builder = MODEL_BUILDERS[MODEL_TYPE]
    
    # Custom builder that incorporates focal loss
    def custom_model_builder(hp):
        # Get the base model
        model = model_builder(hp, num_classes=2)
        
        # Get optimizer with gradient clipping - choose only one clipping method
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hp.Float("learning_rate", min_value=1e-5, max_value=1e-3, sampling="log"),
            clipnorm=1.0  # Keep only this clipping method and remove clipvalue
        )
        
        # Recompile with focal loss and robust optimizer
        model.compile(
            optimizer=optimizer,
            loss=focal_loss(
                gamma=hp.Float("focal_gamma", min_value=0.5, max_value=5.0, step=0.5, default=2.0),
                alpha=hp.Float("focal_alpha", min_value=0.1, max_value=0.9, step=0.1, default=0.25)
            ),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                    tf.keras.metrics.AUC()]
        )
        
        return model

    # 4. Data preprocessing fixes - check for extreme values
    # Check for extreme values in your scaled data
    train_X_stats = {
        'min': np.min(train_X),
        'max': np.max(train_X),
        'mean': np.mean(train_X),
        'std': np.std(train_X)
    }
    print(f"Training data stats after scaling: {train_X_stats}")

    # If values are extreme, consider capping them
    if train_X_stats['max'] > 10 or train_X_stats['min'] < -10:
        print("Warning: Extreme values detected in scaled training data. Capping values...")
        train_X = np.clip(train_X, -10, 10)
        test_X = np.clip(test_X, -10, 10)

    # 5. Updated callbacks with NaN detection
    # Define single consolidated callbacks list
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        nan_callback  # Add NaN detection
    ]

    # Set up the hyperparameter tuner
    tuner = kt.Hyperband(
        hypermodel=custom_model_builder,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        hyperband_iterations=1,
        directory='models_dir',
        project_name=f'{MODEL_TYPE}_Enhanced_Classification',
        overwrite=True,  # Start fresh
        executions_per_trial=1,
    )

    # Start hyperparameter search
    print("Starting hyperparameter search...")

    tuner.search(
        train_X, train_y,
        epochs=10,  # Limited epochs for tuning
        batch_size=64,
        validation_data=(test_X, test_y),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1  # Add verbosity to see progress
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
        callbacks=callbacks,
        class_weight=class_weight_dict
    )
    
    #----------------------- ADD FEATURE IMPORTANCE ANALYSIS -----------------------#
    # After training, analyze feature importance (for interpretability)
    print("\nAnalyzing feature importance using SHAP values...")
    
    # Need to flatten the input for SHAP to understand it
    X_sample = test_X[:100].reshape(100, -1)  # Take a sample for efficiency
    
    try:
        # Create explainer - choose appropriate one based on model type
        if MODEL_TYPE in ["RandomForest", "XGBoost", "GradientBoosting", "LightGBM", "CatBoost"]:
            explainer = shap.TreeExplainer(best_model)
        else:
            # For deep learning models, use DeepExplainer or GradientExplainer
            background = train_X[:100].reshape(100, -1)  # Small background dataset
            explainer = shap.DeepExplainer(best_model, background)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Get feature names
        feature_names = []
        for f in important_features:
            for i in range(n_in):
                feature_names.append(f"{f}_t-{n_in-i}")
        
        # Show top important features based on SHAP values
        shap_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(shap_importance)[-20:]  # Top 20 features
        
        print("\nTop 20 important features based on SHAP values:")
        for i in reversed(top_indices):
            if i < len(feature_names):
                print(f"{feature_names[i]}: {shap_importance[i]:.4f}")
    
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("Consider using simpler feature importance methods for this model type.")
    #----------------------- END OF ADDITIONS -----------------------#
    
except Exception as e:
    print(f"An error occurred during model training: {e}")
    import traceback
    traceback.print_exc()

#%%