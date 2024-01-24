from tensorflow.keras.layers import (
    Conv2D,
    Dropout,
    Dense,
    GRU,
    LSTM,
    SimpleRNN,
    Conv1D,
    ConvLSTM1D,
    ConvLSTM2D,
    SeparableConv1D,
    SeparableConv2D,
    DepthwiseConv2D,
    MultiHeadAttention,
    Attention,
    AdditiveAttention,
    MaxPooling1D
)


def build_Dense_layer(hp):
    '''
    Builds optimal Dense Layer
    Last Dense Layer in MLP units = 1
    Input shape: (batch_size, ..., input_dim)
    Output shape: (batch_size, ..., units)
    '''
    return Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", [  'relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu']),
        use_bias=hp.Boolean("use_bias"),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [  'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [  'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [  'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [  'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [  'max_norm', 'non_neg', 'unit_norm'])
    )

def build_Dropout_layer(hp):
    '''
    Builds Optimal Dropout layer
    No inputs needed besides hp to test different dropout rates
    '''
    return Dropout(
        rate=hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1))

def build_MaxPooling1D_Layer(hp, data_format = 'channel_last'):
    '''
    Inputs:
    If data_format="channels_last": 3D tensor with shape (batch_size, steps, features).
    If data_format="channels_first": 3D tensor with shape (batch_size, features, steps).
    Outputs:
    If data_format="channels_last": 3D tensor with shape (batch_size, downsampled_steps, features).
    If data_format="channels_first": 3D tensor with shape (batch_size, features, downsampled_steps).
    '''
    # MaxPooling layer
    return MaxPooling1D(
        pool_size=hp.Int("pool_size", min_value=1, max_value=3, step=1),
        strides=hp.Int("strides", min_value=1, max_value=3, step=1),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format
        )

def build_SimpleRNN_layer(hp, return_sequences=False, seed = 42):
    '''
    Builds Optimal SimpleRNN using Keras HyperParameter Tuner
    Inupts hp for keras hyperparameter tuning, return_sequences, and seed
    Set return_sequences = True if layer is NOT last in MLP
    Includes Dropout testing so no need in adding Dropout Layer
    Input Shape: (num_training_examples, num_timesteps, num_features)
    '''
    return SimpleRNN(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", ['tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu']),
        use_bias=hp.Boolean("use_bias"),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'orthogonal', 'identity', 'lecun_normal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'orthogonal', 'identity', 'lecun_normal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [  'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [  'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [  'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [  'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [  'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [  'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [  'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll"),
        seed = seed
    )

def build_LSTM_layer(hp, return_sequences=False, seed = 42):
    '''
    Builds Optimal LSTM Layer using Keras Hyperparamter Tuner
    Inupts hp for keras hyperparameter tuning, return_sequences, and seed
    Set return_sequences = True if layer is NOT last in MLP
    Includes Dropout testing so no need in adding Dropout Layer
    Input shape: (batch, timesteps, feature)
    '''
    return LSTM(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", ['l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", ['l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", ['l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", ['l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        seed = seed,
        return_sequences=return_sequences, 
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )

def build_GRU_layer(hp, return_sequences=False, seed = 42):
    '''
    Builds Optimal GRU Layer using Keras Hyperparamter Tuner
    Inupts hp for keras hyperparameter tuning, return_sequences, seed, and reset after
    Set return_sequences = True if layer is NOT last in MLP
    Includes Dropout testing so no need in adding Dropout Layer
    reset_after: whether to apply reset gate after or before matrix multiplication False is "before", True is "after"
    Input shape: (batch, timesteps, feature)
    '''
    return GRU(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", ['l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", ['l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", ['l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", ['l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", ['max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        seed = seed,
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll"),
        reset_after = hp.Boolean("reset_after")
    )

def build_Conv1D_layer(hp, data_format = 'channels_last'):
    '''
    data_format = "channels_last" corresponds to inputs with shape (batch, steps, features)
    data_format = "channels_first" corresponds to inputs with shape (batch, features, steps).
    Input shape: (batch, steps, channels)
    Output Shape: (batch, new_steps, filters)
    '''
    return Conv1D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=["l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=["l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=["l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=["max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=["max_norm", "non_neg", "unit_norm"])
    )
    
def build_Conv2D_layer(hp, data_format = 'channels_last'):
    '''
    Builds Optimal Conv2D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    data_format = "channels_last" corresponds to inputs with shape (batch, steps, features)
    data_format = "channels_first" corresponds to inputs with shape (batch, features, steps).
    Outputs:
    data_format="channels_last": tensor with shape: (batch_size, new_height, new_width, filters)
    data_format="channels_first": tensor with shape: (batch_size, filters, new_height, new_width)
    '''
    return Conv2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=["l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=["l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=["l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=["max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=["max_norm", "non_neg", "unit_norm"])
    )

def build_ConvLSTM1D_layer(hp, return_sequences=False, data_format = 'channels_last', seed = 42):
    '''
    Builds Optimal ConvLSTM1D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    Set return_sequences = True if NOT last layer in MLP
    data_format = "channels_last" corresponds to inputs with shape (samples, time, rows, channels)
    data_format = "channels_first" corresponds to inputs with shape (samples, time, channels, rows)
    Outputs:
    If return_state: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each 3D tensor with shape: (samples, filters, new_rows) if data_format='channels_first' or shape: (samples, new_rows, filters) if data_format='channels_last'. rows values might have changed due to padding.
    If return_sequences: 4D tensor with shape: (samples, timesteps, filters, new_rows) if data_format='channels_first' or shape: (samples, timesteps, new_rows, filters) if data_format='channels_last'.
    Else, 3D tensor with shape: (samples, filters, new_rows) if data_format='channels_first' or shape: (samples, new_rows, filters) if data_format='channels_last'.
    '''
    return ConvLSTM1D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        activation=hp.Choice("activation", values=["tanh", "sigmoid", "relu"]),
        recurrent_activation=hp.Choice("recurrent_activation", values=["sigmoid", "tanh", "relu"]),
        use_bias=hp.Boolean("use_bias", default=True),
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        recurrent_initializer=hp.Choice("recurrent_initializer", values=["orthogonal", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        unit_forget_bias=hp.Boolean("unit_forget_bias", default=True),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        recurrent_constraint=hp.Choice("recurrent_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.1),
        seed=seed,
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful")
    )

def build_ConvLSTM2D_layer(hp, return_sequences=False, data_format = 'channels_last', seed = 42):
    '''
    Builds Optimal ConvLSTM2D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    Set return_sequences = True if NOT last layer in MLP
    data_format = "channels_last" corresponds to inputs with shape (samples, time, rows, cols, channels)
    data_format = "channels_first" corresponds to inputs with shape (samples, time, channels, rows, cols)
    Outputs:
    If return_state: a list of tensors. The first tensor is the output. The remaining tensors are the last states, each 4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'. rows and cols values might have changed due to padding.
    If return_sequences: 5D tensor with shape: (samples, timesteps, filters, new_rows, new_cols) if data_format='channels_first' or shape: (samples, timesteps, new_rows, new_cols, filters) if data_format='channels_last'.
    Else, 4D tensor with shape: (samples, filters, new_rows, new_cols) if data_format='channels_first' or shape: (samples, new_rows, new_cols, filters) if data_format='channels_last'.
    '''
    return ConvLSTM2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        activation=hp.Choice("activation", values=["tanh", "sigmoid", "relu"]),
        recurrent_activation=hp.Choice("recurrent_activation", values=["sigmoid", "tanh", "relu"]),
        use_bias=hp.Boolean("use_bias", default=True),
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        recurrent_initializer=hp.Choice("recurrent_initializer", values=["orthogonal", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        unit_forget_bias=hp.Boolean("unit_forget_bias", default=True),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        recurrent_constraint=hp.Choice("recurrent_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.1),
        seed=seed,
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful")
    )

def build_SeparableConv1D_layer(hp, data_format = 'channels_last',):
    '''
    Builds Optimal SeparableConv1D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    If data_format="channels_last": A 3D tensor with shape: (batch_shape, steps, channels)
    If data_format="channels_first": A 3D tensor with shape: (batch_shape, channels, steps)
    Outputs:
    If data_format="channels_last": A 3D tensor with shape: (batch_shape, new_steps, filters)
    If data_format="channels_first": A 3D tensor with shape: (batch_shape, filters, new_steps)
    '''
    return SeparableConv1D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        depth_multiplier=hp.Int("depth_multiplier", min_value=1, max_value=3),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        depthwise_initializer=hp.Choice("depthwise_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        pointwise_initializer=hp.Choice("pointwise_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        depthwise_regularizer=hp.Choice("depthwise_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        pointwise_regularizer=hp.Choice("pointwise_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        depthwise_constraint=hp.Choice("depthwise_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        pointwise_constraint=hp.Choice("pointwise_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"])
    )

def build_SeparableConv2D_layer(hp, data_format = 'channels_last'):
    '''
    Builds Optimal SeparableConv2D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    If data_format="channels_last": A 4D tensor with shape: (batch_size, height, width, channels)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, channels, height, width)
    Outputs:
    If data_format="channels_last": A 4D tensor with shape: (batch_size, new_height, new_width, filters)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, filters, new_height, new_width)
    '''
    return SeparableConv2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
        data_format = data_format,
        dilation_rate=hp.Int("dilation_rate", min_value=1, max_value=4),
        depth_multiplier=hp.Int("depth_multiplier", min_value=1, max_value=3),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        depthwise_initializer=hp.Choice("depthwise_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        pointwise_initializer=hp.Choice("pointwise_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        depthwise_regularizer=hp.Choice("depthwise_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        pointwise_regularizer=hp.Choice("pointwise_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        depthwise_constraint=hp.Choice("depthwise_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        pointwise_constraint=hp.Choice("pointwise_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"])
    )

def build_DepthwiseConv2D_layer(hp, data_format = 'channels_last'):
    '''
    Builds Optimal SeparableConv2D layer using Keras Hyperparameter tuner
    Inputs:
    hp (Keras Hyperparamter tuner)
    If data_format="channels_last": A 4D tensor with shape: (batch_size, height, width, channels)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, channels, height, width)
    Outputs:
    If data_format="channels_last": A 4D tensor with shape: (batch_size, new_height, new_width, channels * depth_multiplier)
    If data_format="channels_first": A 4D tensor with shape: (batch_size, channels * depth_multiplier, new_height, new_width)
    Returns:
    A 4D tensor representing activation(depthwise_conv2d(inputs, kernel) + bias).
    ValueError: when both strides > 1 and dilation_rate > 1.
    '''
    return DepthwiseConv2D(
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=(1, 1),
        padding=hp.Choice("padding", values=["valid", "same"]),
        depth_multiplier=hp.Int("depth_multiplier", min_value=1, max_value=3),
        data_format=data_format,
        dilation_rate=(1, 1),
        activation=hp.Choice("activation", values=["relu", "sigmoid", "tanh"]),
        use_bias=hp.Boolean("use_bias", default=True),
        depthwise_initializer=hp.Choice("depthwise_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        depthwise_regularizer=hp.Choice("depthwise_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        depthwise_constraint=hp.Choice("depthwise_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"])
    )

def build_MultiHeadAttention_layer(hp):
    return MultiHeadAttention(
        num_heads=hp.Int("num_heads", min_value=1, max_value=8, step=1),
        key_dim=hp.Int("key_dim", min_value=32, max_value=256, step=32),
        value_dim=hp.Int("value_dim", min_value=32, max_value=256, step=32),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        use_bias=hp.Boolean("use_bias", default=True),
        output_shape=None,
        attention_axes=None,
        kernel_initializer=hp.Choice("kernel_initializer", values=["glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        bias_initializer=hp.Choice("bias_initializer", values=["zeros", "ones", "glorot_uniform", "glorot_normal", "he_uniform", "he_normal"]),
        kernel_regularizer=hp.Choice("kernel_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        bias_regularizer=hp.Choice("bias_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        activity_regularizer=hp.Choice("activity_regularizer", values=[None, "l1", "l2", "l1_l2"]),
        kernel_constraint=hp.Choice("kernel_constraint", values=[None, "max_norm", "non_neg", "unit_norm"]),
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"])
    )

def build_Attention_layer(hp):
    return Attention(
        use_scale=hp.Boolean("use_scale", default=False),
        score_mode=hp.Choice("score_mode", values=["dot", "scaled_dot", "additive"]),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        seed=hp.Int("seed", min_value=1, max_value=100)
    )

def build_AdditiveAttention_layer(hp):
    return AdditiveAttention(
        use_scale=hp.Boolean("use_scale", default=True),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05)
    )