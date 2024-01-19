from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# Function to build Conv1D layer with hyperparameter tuning
def build_conv2d_layer(hp, input_shape):
    return Conv2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
    )

# Build the model with Conv1D layer and hyperparameter tuning
def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[60, 1]))
    
    for rate in range(1, 9):
        model.add(build_conv2d_layer(hp, input_shape=(60, 1)))
    
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss="mse", optimizer="adam")
    return model

# Instantiate the tuner
tuner = RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=5,  # Number of hyperparameter combinations to try
    executions_per_trial=1,
    directory="my_dir",
    project_name="conv1d_tuning",
)

# Perform the hyperparameter tuning
tuner.search_space_summary()
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
