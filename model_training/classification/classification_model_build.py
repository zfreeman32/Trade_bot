from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Concatenate, Add, Activation
from keras.optimizers import Adam
from model_layers import (build_Attention_layer, build_SeparableConv1D_layer, 
                         build_MultiHeadAttention_layer, build_Dense_layer, build_LSTM_layer, 
                         build_GRU_layer, build_SimpleRNN_layer, build_Conv1D_layer, 
                         build_Dropout_layer, build_MaxPooling1D_Layer)
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

seed = 42

#%%
#----------------------- MODEL EVAL METRICS -----------------------#
# ADD MODEL EVAL METRICS
def directional_accuracy(y_true, y_pred):
    """Measures if the prediction is directionally correct (up/down)"""
    # Convert probabilities to binary
    pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    true_binary = tf.cast(y_true, tf.float32)
    
    # Calculate accuracy
    correct_predictions = tf.equal(pred_binary, true_binary)
    return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def precision_recall_f1(y_true, y_pred):
    """Calculate precision, recall and F1 all at once"""
    # Convert probabilities to binary
    pred_binary = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
    true_binary = tf.cast(y_true, tf.float32)
    
    # Calculate metrics
    true_positives = tf.reduce_sum(true_binary * pred_binary)
    predicted_positives = tf.reduce_sum(pred_binary)
    actual_positives = tf.reduce_sum(true_binary)
    
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    recall = true_positives / (actual_positives + tf.keras.backend.epsilon())
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    
    return precision, recall, f1

metrics=['accuracy', directional_accuracy, tf.keras.metrics.AUC(),
         tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

def build_LSTM_classifier(hp, num_classes=2):
    """
    LSTM-based classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add LSTM layers based on the hyperparameters
    for i in range(hp.Int("num_LSTM_layers", min_value=1, max_value=3, step=1)):
        model.add(build_LSTM_layer(hp, return_sequences=True))
        
    # Add last LSTM layer
    model.add(build_LSTM_layer(hp, return_sequences=False))
    
    # Add Dense layers based on the hyperparameters
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_GRU_classifier(hp, num_classes=2):
    """
    GRU-based classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add GRU layers based on the hyperparameters
    for i in range(hp.Int("num_GRU_layers", min_value=1, max_value=3, step=1)):
        model.add(build_GRU_layer(hp, return_sequences=True))
        
    # Add last GRU layer
    model.add(build_GRU_layer(hp, return_sequences=False))
    
    # Add Dense layers based on the hyperparameters
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_Conv1D_classifier(hp, num_classes=2, data_format='channels_last'):
    """
    1D Convolutional classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add Conv1D layers
    for i in range(hp.Int("num_conv1d_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Conv1D_layer(hp, data_format=data_format))
        if hp.Boolean(f"add_pooling_{i}", default=True):
            model.add(build_MaxPooling1D_Layer(hp, data_format=data_format))
    
    # Flatten layer
    model.add(Flatten())
    
    # Add Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_Conv1D_LSTM_classifier(hp, num_classes=2, data_format='channels_last'):
    """
    Hybrid CNN-LSTM classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add Conv1D layers
    for i in range(hp.Int("num_conv_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Conv1D_layer(hp, data_format=data_format))
        model.add(build_MaxPooling1D_Layer(hp, data_format=data_format))
    
    # Important: don't flatten before LSTM
    # Add LSTM layers
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = (i < hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1) - 1)
        model.add(build_LSTM_layer(hp, return_sequences=return_sequences))
    
    # Add Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=2, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_BiLSTM_Attention_classifier(hp, num_classes=2):
    """
    Bidirectional LSTM classifier with Attention mechanism
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add bidirectional LSTM layers
    for i in range(hp.Int("num_bilstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = True if i < hp.Int("num_bilstm_layers", min_value=1, max_value=3, step=1) - 1 or i == 0 else False
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hp.Int(f"lstm_units_{i}", min_value=32, max_value=128, step=32),
                activation=hp.Choice(f"activation_{i}", ['relu', 'tanh', 'sigmoid']),
                return_sequences=return_sequences,
                dropout=hp.Float(f"dropout_{i}", min_value=0.0, max_value=0.5, step=0.1),
                recurrent_dropout=hp.Float(f"rec_dropout_{i}", min_value=0.0, max_value=0.5, step=0.1)
            )
        ))
        
        # Add layer normalization
        model.add(tf.keras.layers.LayerNormalization())
    
    # Add attention layer
    model.add(build_Attention_layer(hp))
    
    # Add Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_Transformer_classifier(hp, num_classes=2, data_format='channels_last'):
    """
    Transformer-based classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    model = Sequential()
    
    # Add positional encoding layer
    model.add(tf.keras.layers.Lambda(
        lambda x: x + tf.cast(tf.math.sin(
            tf.range(tf.shape(x)[1], dtype=tf.float32)[None, :, None] * 
            (1000.0 ** (-tf.range(0, tf.shape(x)[2], dtype=tf.float32)[None, None, :] / tf.shape(x)[2]))
        ), dtype=tf.float32),
        input_shape=(None, None)
    ))
    
    # Add transformer blocks
    for i in range(hp.Int("num_transformer_blocks", min_value=1, max_value=4, step=1)):
        # Multi-head attention
        model.add(build_MultiHeadAttention_layer(hp))
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
        
        # Feed-forward network
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
        model.add(tf.keras.layers.LayerNormalization(epsilon=1e-6))
    
    # Global average pooling
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    
    # Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        model.add(build_Dense_layer(hp))
        model.add(build_Dropout_layer(hp))
    
    # Output layer with appropriate activation
    if num_classes == 2:
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
        
    else:
        model.add(Dense(num_classes, activation='softmax'))
        loss = 'categorical_crossentropy'
        
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_MultiStream_classifier(hp, input_shape = (240, 1), num_classes=2, data_format='channels_last'):
    """
    Multi-stream hybrid classifier combining CNN, LSTM, and Attention
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Stream 1: Convolutional stream
    conv_stream = input_layer
    for i in range(hp.Int("num_conv_layers", min_value=1, max_value=3, step=1)):
        conv_stream = build_Conv1D_layer(hp, data_format=data_format)(conv_stream)
        conv_stream = build_MaxPooling1D_Layer(hp, data_format=data_format)(conv_stream)
    
    # Stream 2: LSTM stream
    lstm_stream = input_layer
    for i in range(hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1)):
        return_sequences = True if i < hp.Int("num_lstm_layers", min_value=1, max_value=3, step=1) - 1 else False
        lstm_stream = build_LSTM_layer(hp, return_sequences=return_sequences)(lstm_stream)
    
    # Stream 3: Attention stream
    attention_stream = input_layer
    attention_stream = build_MultiHeadAttention_layer(hp)(
        attention_stream, attention_stream, attention_stream
    )
    
    # Flatten all streams
    conv_stream = Flatten()(conv_stream)
    if hp.Choice("flatten_lstm", values=[True, False]):
        lstm_stream = Flatten()(lstm_stream)
    attention_stream = Flatten()(attention_stream)
    
    # Concatenate the streams
    merged = tf.keras.layers.Concatenate()([conv_stream, lstm_stream, attention_stream])
    
    # Dense layers after merge
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        merged = build_Dense_layer(hp)(merged)
        merged = build_Dropout_layer(hp)(merged)
    
    # Output layer with appropriate activation
    if num_classes == 2:
        output_layer = Dense(1, activation='sigmoid')(merged)
        loss = 'binary_crossentropy'
        
    else:
        output_layer = Dense(num_classes, activation='softmax')(merged)
        loss = 'categorical_crossentropy'
        
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_ResNet_classifier(hp, input_shape = (240, 1), num_classes=2, data_format='channels_last'):
    """
    ResNet-inspired classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    # Initial convolution
    x = build_Conv1D_layer(hp, data_format=data_format)(input_layer)
    
    # ResNet blocks
    for i in range(hp.Int("num_res_blocks", min_value=1, max_value=6, step=1)):
        # Store the input to the block for skip connection
        block_input = x
        
        # First conv layer in block
        x = build_SeparableConv1D_layer(hp, data_format=data_format)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        # Second conv layer in block
        x = build_SeparableConv1D_layer(hp, data_format=data_format)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Skip connection with dimension matching if needed
        if block_input.shape[-1] != x.shape[-1] or block_input.shape[-2] != x.shape[-2]:
            block_input = tf.keras.layers.Conv1D(
                filters=x.shape[-1], 
                kernel_size=1, 
                padding='same',
                data_format=data_format
            )(block_input)
        
        # Add skip connection
        x = Add()([x, block_input])
        x = Activation('relu')(x)
        
        # Optional pooling
        if hp.Choice(f"pool_after_block_{i}", values=[True, False]):
            x = build_MaxPooling1D_Layer(hp, data_format=data_format)(x)
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        x = build_Dense_layer(hp)(x)
        x = build_Dropout_layer(hp)(x)
    
    # Output layer with appropriate activation
    if num_classes == 2:
        output_layer = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        
    else:
        output_layer = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_TCN_classifier(hp, input_shape = (240, 1), num_classes=2, data_format='channels_last'):
    """
    Temporal Convolutional Network (TCN) classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    - num_classes: Number of classes (2 for binary, >2 for multi-class)
    - data_format: Format of input data ('channels_last' or 'channels_first')
    
    Returns:
    - Compiled classification model
    """
    # Define the input
    input_layer = tf.keras.layers.Input(shape=input_shape)
    
    x = input_layer
    n_filters = hp.Int("n_filters", min_value=32, max_value=128, step=32)
    
    # TCN blocks with increasing dilation rates
    for i in range(hp.Int("num_tcn_blocks", min_value=1, max_value=6, step=1)):
        dilation_rate = 2**i  # Exponentially increasing dilation
        
        # First dilated conv
        conv1 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu',
            data_format=data_format
        )(x)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = build_Dropout_layer(hp)(conv1)
        
        # Second dilated conv
        conv2 = tf.keras.layers.Conv1D(
            filters=n_filters,
            kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
            padding='causal',
            dilation_rate=dilation_rate,
            activation='relu',
            data_format=data_format
        )(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = build_Dropout_layer(hp)(conv2)
        
        # Skip connection with dimension matching if needed
        if x.shape[-1] != n_filters:
            x = tf.keras.layers.Conv1D(
                filters=n_filters,
                kernel_size=1,
                padding='same',
                data_format=data_format
            )(x)
        
        x = Add()([x, conv2])
    
    # Global pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    for i in range(hp.Int("num_dense_layers", min_value=1, max_value=3, step=1)):
        x = build_Dense_layer(hp)(x)
        x = build_Dropout_layer(hp)(x)
    
    # Output layer with appropriate activation
    if num_classes == 2:
        output_layer = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        
    else:
        output_layer = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
        
    
    # Create and compile the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss=loss,
        metrics=metrics
    )
    
    return model

def build_RandomForestClassifier_model(hp):
    """
    RandomForest classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    
    Returns:
    - RandomForestClassifier model
    """
    model = RandomForestClassifier(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        min_samples_split=hp.Int("min_samples_split", min_value=2, max_value=10, step=2),
        min_samples_leaf=hp.Int("min_samples_leaf", min_value=1, max_value=5, step=1),
        max_features=hp.Choice("max_features", ["auto", "sqrt", "log2"]),
        class_weight=hp.Choice("class_weight", [None, "balanced", "balanced_subsample"]),
        random_state=seed,
        n_jobs=-1
    )
    return model

def build_XGBoostClassifier_model(hp):
    """
    XGBoost classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    
    Returns:
    - XGBClassifier model
    """
    model = XGBClassifier(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        subsample=hp.Float("subsample", min_value=0.5, max_value=1.0, step=0.1),
        colsample_bytree=hp.Float("colsample_bytree", min_value=0.5, max_value=1.0, step=0.1),
        gamma=hp.Float("gamma", min_value=0, max_value=5, step=0.5),
        scale_pos_weight=hp.Float("scale_pos_weight", min_value=1, max_value=10, step=1),
        random_state=seed,
        use_label_encoder=False,
        n_jobs=-1
    )
    return model

def build_LightGBMClassifier_model(hp):
    """
    LightGBM classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    
    Returns:
    - LGBMClassifier model
    """
    model = LGBMClassifier(
        n_estimators=hp.Int("n_estimators", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        num_leaves=hp.Int("num_leaves", min_value=20, max_value=150, step=10),
        max_depth=hp.Int("max_depth", min_value=3, max_value=15, step=3),
        subsample=hp.Float("subsample", min_value=0.5, max_value=1.0, step=0.1),
        colsample_bytree=hp.Float("colsample_bytree", min_value=0.5, max_value=1.0, step=0.1),
        class_weight=hp.Choice("class_weight", [None, "balanced"]),
        random_state=seed,
        n_jobs=-1
    )
    return model

def build_CatBoostClassifier_model(hp):
    """
    CatBoost classifier for time-series data
    
    Parameters:
    - hp: Keras tuner hyperparameters
    
    Returns:
    - CatBoostClassifier model
    """
    model = CatBoostClassifier(
        iterations=hp.Int("iterations", min_value=50, max_value=300, step=50),
        learning_rate=hp.Float("learning_rate", min_value=0.01, max_value=0.3, step=0.05),
        depth=hp.Int("depth", min_value=3, max_value=15, step=3),
        l2_leaf_reg=hp.Float("l2_leaf_reg", min_value=1, max_value=10, step=1),
        random_state=seed,
        verbose=0,
        auto_class_weights=hp.Choice("auto_class_weights", [None, "Balanced", "SqrtBalanced"])
    )
    return model

# Example usage:
# For binary classification:
# tuner = kt.Hyperband(
#     hypermodel=lambda hp: build_LSTM_classifier(hp, num_classes=2),
#     objective='val_accuracy',
#     max_epochs=100,
#     factor=3,
#     hyperband_iterations=1,
#     directory='models_dir',
#     project_name='LSTM_binary_classification'
# )

# For multi-class classification:
# tuner = kt.Hyperband(
#     hypermodel=lambda hp: build_LSTM_classifier(hp, num_classes=5),  # for 5 classes
#     objective='val_accuracy',
#     max_epochs=100,
#     factor=3,
#     hyperband_iterations=1,
#     directory='models_dir',
#     project_name='LSTM_multiclass_classification'
# )