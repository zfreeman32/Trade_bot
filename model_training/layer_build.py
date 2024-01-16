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
    AdditiveAttention
)


def build_Dense_layer(hp, input_shape, return_sequences=False):
    return Dense(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", [None, 'relu', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu']),
        use_bias=hp.Boolean("use_bias"),
        return_sequences=return_sequences,
        input_shape=(input_shape),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm'])
    )

def build_Dropout_layer(hp):
    return Dropout(
        rate=hp.Float('dropout_rate', min_value=0, max_value=0.5, step=0.1))

def build_SimpleRNN_layer(hp, input_shape, return_sequences=False):
    return SimpleRNN(
        units=hp.Int("units", min_value=32, max_value=512, step=32),
        activation=hp.Choice("activation", ['tanh', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu']),
        use_bias=hp.Boolean("use_bias"),
        return_sequences=return_sequences,
        input_shape=(input_shape),
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'orthogonal', 'identity', 'lecun_normal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'orthogonal', 'identity', 'lecun_normal', 'lecun_uniform', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_sequences=hp.Boolean("return_sequences"),
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )

def build_LSTM_layer(hp, input_shape, return_sequences=False):
    return LSTM(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        return_sequences=True, 
        input_shape=(input_shape),
        return_sequences=return_sequences,
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )

def build_GRU_layer(hp, input_shape, return_sequences=False):
    return GRU(
        units=hp.Int("units_first", min_value=32, max_value=512, step=32), 
        return_sequences=True, 
        input_shape=(input_shape),
        return_sequences=return_sequences,
        activation=hp.Choice("activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        recurrent_activation=hp.Choice("recurrent_activation", ['tanh', 'relu', 'log_softmax', 'softmax', 'softplus', 'softsign', 'elu', 'exponential', 'linear', 'relu6', 'gelu' ]),
        use_bias=True,
        kernel_initializer=hp.Choice("kernel_initializer", ['zeros','ones','constant','random_normal','random_uniform','truncated_normal','glorot_normal','glorot_uniform','he_normal','he_uniform','lecun_normal','lecun_uniform']),
        recurrent_initializer=hp.Choice("recurrent_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        bias_initializer=hp.Choice("bias_initializer", ['zeros','ones','constant','random_normal','random_uniform','orthogonal','identity','lecun_normal','lecun_uniform','glorot_normal','glorot_uniform','he_normal','he_uniform']),
        unit_forget_bias = hp.Boolean("forget_bias"),
        kernel_regularizer=hp.Choice("kernel_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        recurrent_regularizer=hp.Choice("recurrent_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        bias_regularizer=hp.Choice("bias_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        activity_regularizer=hp.Choice("activity_regularizer", [None, 'l1', 'l2', 'l1_l2']),
        kernel_constraint=hp.Choice("kernel_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        recurrent_constraint=hp.Choice("recurrent_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        bias_constraint=hp.Choice("bias_constraint", [None, 'max_norm', 'non_neg', 'unit_norm']),
        dropout=hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.05),
        recurrent_dropout=hp.Float("recurrent_dropout", min_value=0.0, max_value=0.5, step=0.05),
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful"),
        unroll=hp.Boolean("unroll")
    )

def build_Conv1D_layer(hp, input_shape):
    return Conv1D(
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
    
def build_Conv2D_layer(hp, input_shape):
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
        bias_constraint=hp.Choice("bias_constraint", values=[None, "max_norm", "non_neg", "unit_norm"])
    )

def build_ConvLSTM1D_layer(hp, return_sequences=False):
    return ConvLSTM1D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
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
        seed=hp.Int("seed", min_value=0, max_value=999),
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful")
    )

def build_ConvLSTM2D_layer(hp, return_sequences=False):
    return ConvLSTM2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
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
        seed=hp.Int("seed", min_value=0, max_value=999),
        return_sequences=return_sequences,
        return_state=hp.Boolean("return_state"),
        go_backwards=hp.Boolean("go_backwards"),
        stateful=hp.Boolean("stateful")
    )

def build_SeparableConv1D_layer(hp):
    return SeparableConv1D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
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

def build_SeparableConv2D_layer(hp):
    return SeparableConv2D(
        filters=hp.Int("filters", min_value=16, max_value=64, step=16),
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=hp.Int("strides", min_value=1, max_value=3),
        padding=hp.Choice("padding", values=["valid", "same"]),
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

def build_DepthwiseConv2D_layer(hp):
    return DepthwiseConv2D(
        kernel_size=hp.Int("kernel_size", min_value=2, max_value=5),
        strides=(1, 1),
        padding=hp.Choice("padding", values=["valid", "same"]),
        depth_multiplier=hp.Int("depth_multiplier", min_value=1, max_value=3),
        data_format=hp.Choice("data_format", values=[None, "channels_last", "channels_first"]),
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