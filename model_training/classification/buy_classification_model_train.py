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
from numpy.lib.stride_tricks import sliding_window_view
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

# %% Configuration
# Select which model to use - change this to try different models
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
# Split features and target
features = data.drop(columns=['long_signal', 'short_signal', 'close_position'])  # All features except targets
target = data[['long_signal']]  # Target is 'buy_signal'

# Ensure target is categorical (0 or 1)
target = target.astype(int)

# Configuration for window size
n_in = 240  # Number of past observations (lookback window)
n_features = features.shape[1]  # Number of feature columns

print(f"Features shape: {features.shape}, Target shape: {target.shape}")
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

# Apply standardization to each feature across the time dimension
# Reshape to 2D for scaling
train_X_2d = train_X.reshape(-1, n_features)
scaler = RobustScaler()
train_X_2d = scaler.fit_transform(train_X_2d)
train_X = train_X_2d.reshape(train_X.shape)

# Apply same scaling to test data
test_X_2d = test_X.reshape(-1, n_features)
test_X_2d = scaler.transform(test_X_2d)
test_X = test_X_2d.reshape(test_X.shape)

# Flatten target arrays
train_y = train_y.flatten()
test_y = test_y.flatten()

print(f"Train X shape: {train_X.shape}")
print(f"Test X shape: {test_X.shape}")
print(f"Train y shape: {train_y.shape}")
print(f"Test y shape: {test_y.shape}")

# Check for class imbalance
train_class_counts = np.bincount(train_y.astype(int))
print(f"Class distribution in training set: {train_class_counts}")
print(f"Class imbalance ratio: 1:{train_class_counts[0]/train_class_counts[1]:.2f}")

# Compute class weights for imbalanced classes
class_weight = {
    0: 1.0,
    1: train_class_counts[0] / train_class_counts[1]  # Weight the minority class more
}
print(f"Using class weights: {class_weight}")

# %%
# Set up the model tuner
print(f"Setting up {MODEL_TYPE} model for hyperparameter tuning...")

try:
    # Create the tuner using the selected model
    model_builder = MODEL_BUILDERS[MODEL_TYPE]
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, num_classes=2),  # Binary classification
        objective='val_accuracy',
        max_epochs=100,
        factor=3,
        hyperband_iterations=1,
        directory='models_dir',
        project_name=f'{MODEL_TYPE}_Classification_Tuning'
    )

    # Create callbacks for early stopping
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
        )
    ]

    # Perform hyperparameter search
    print("Starting hyperparameter search...")
    tuner.search(
        train_X, train_y,
        epochs=10,  # Limited epochs for tuning
        batch_size=64,
        validation_data=(test_X, test_y),
        callbacks=callbacks,
        class_weight=class_weight
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
        class_weight=class_weight
    )

    # Plot training history
    print("Plotting training history...")
    pyplot.figure(figsize=(12, 10))
    
    # Plot accuracy
    pyplot.subplot(2, 2, 1)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='validation')
    pyplot.title('Model Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    
    # Plot loss
    pyplot.subplot(2, 2, 2)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='validation')
    pyplot.title('Model Loss')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()

    # Generate predictions
    print("Generating predictions on test data...")
    y_pred_prob = best_model.predict(test_X)
    
    # Handle different output formats based on model type
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
        # Multi-class case
        y_pred = np.argmax(y_pred_prob, axis=1)
    else:
        # Binary case - get probabilities and threshold
        y_pred_prob = y_pred_prob.flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Plot confusion matrix
    pyplot.subplot(2, 2, 3)
    cm = confusion_matrix(test_y, y_pred)
    pyplot.imshow(cm, interpolation='nearest', cmap=pyplot.cm.Blues)
    pyplot.title('Confusion Matrix')
    pyplot.colorbar()
    tick_marks = np.arange(2)
    pyplot.xticks(tick_marks, ['Negative', 'Positive'])
    pyplot.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pyplot.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    
    # Plot ROC curve
    pyplot.subplot(2, 2, 4)
    fpr, tpr, _ = roc_curve(test_y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    pyplot.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    pyplot.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.05])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Receiver Operating Characteristic')
    pyplot.legend(loc="lower right")
    
    pyplot.tight_layout()
    pyplot.savefig(f'{MODEL_TYPE}_classification_results.png')
    pyplot.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_y, y_pred))
    
    # Save the model
    model_path = f'models/{MODEL_TYPE}_classifier.h5'
    best_model.save(model_path)
    print(f"Model saved to {model_path}")

except Exception as e:
    print(f"An error occurred during model training: {e}")
    import traceback
    traceback.print_exc()

print("Training process completed.")