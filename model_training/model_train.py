# %%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
from pandas import DataFrame, concat
from math import sqrt
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import keras_tuner as kt
from features.all_indicators import generate_all_indicators
from model_training import model_build
from features import call_Strategies
from data import preprocess_data

# Function to frame the dataset as supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Combine all
    agg = concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# process data
# Generate indicators
def preprocess_data(data):
    df_with_indicators = generate_all_indicators(data)

    # Generate Signals
    df_strategies = pd.DataFrame(data).reset_index(drop=True)
    print(df_strategies.head())

    # Now pass the DataFrame to the generate_all_signals function
    df_strategies = call_Strategies.generate_all_signals(df_strategies)
    df_strategies = df_strategies.drop(['kama_signal'],axis=1)

    # Convert columns to string
    data['Date'] = data['Date'].astype(str)  
    data['Time'] = data['Time'].astype(str)
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

    # Identify common columns except 'Date'
    common_cols = [col for col in df_strategies.columns if col in df_with_indicators.columns and col != 'Date']

    # Remove common columns from df2 before merging
    df_with_indicators = df_with_indicators.drop(columns=common_cols)

    # Then do your merge
    df_strategies = df_strategies.set_index('Date')
    df_with_indicators = df_with_indicators.set_index('Date')
    merged_df = pd.concat([df_strategies, df_with_indicators], axis=1).reset_index()
    df = pd.DataFrame(merged_df)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')  # Convert to datetime
    df['Minutes'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 60  
    df.drop(columns=['Time'], inplace=True)

    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Remove 'Time' and 'Date' from non_numeric_cols if they exist
    cols_to_encode = [col for col in non_numeric_cols if col not in ['Time']]

    # Initialize LabelEncoder
    label_encoders = {}
    for col in cols_to_encode:
        # Replace NaN values with 0
        df[col].fillna(0, inplace=True)
        df[col].replace("", 0, inplace=True)
        df[col] = df[col].fillna('').apply(str).str.strip() 
        print(df[col].shape) # Convert all values to string and strip whitespace
        le = LabelEncoder()
        df[col]=df[col].astype(str)
        df[col] = le.fit_transform(df[col]) 
        label_encoders[col] = le 

    df.fillna(-9999, inplace=True)
    return df

#%% regression target and features
# Load EURUSD 1-minute data
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled_EURUSD_1min.csv'
data = pd.read_csv(
    file_path, 
    header=0
)
data = data.tail(1000)

df = preprocess_data(data)
# Exclude 'Close' column from features
features = df.drop(columns=['Close'])  
target = df[['Close']]  # Keeping 'Close' as target

# Frame the dataset for multi-step forecasting
n_in = 240  # Number of past observations (1 Month)
n_out = 15  # Number of future observations to forecast (1 Day)
supervised_features = series_to_supervised(features.values, n_in=n_in, n_out=n_out)
supervised_target = series_to_supervised(target.values, n_in=n_in, n_out=n_out)

# Extract input (X) and output (y)
n_features = features.shape[1]  # Number of features excluding 'Close'
n_obs = n_in * n_features  # Total past observations used as input

X = supervised_features.iloc[:, :n_obs].values  # Features (past observations)
y = supervised_target.iloc[:, n_in * target.shape[1]:].values  # Target ('Close' values)

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

# Splitting the dataset into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape X for Conv1D input (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], n_in, -1))  # Assuming n_in=60
test_X = test_X.reshape((test_X.shape[0], n_in, -1))

print("Train (X) shape:", train_X.shape)
print("Test (X) shape:", test_X.shape)
print("Train (Y) shape:", train_y.shape)
print("Test (Y) shape:", test_y.shape)
print(df.dtypes[df.dtypes == 'object'])
df_nans = df.isna().sum()  # Check for NaNs
print(df.applymap(lambda x: isinstance(x, str)).sum())  # Check if any strings remain
print(train_X.dtype, test_X.dtype)  

# Hyperparameter tuning using RandomSearch from Kerastuner
tuner = kt.Hyperband(
    hypermodel=lambda hp: model_build.build_LSTM_model(hp),
    objective='accuracy',
    max_epochs=100,
    factor=3,
    hyperband_iterations=1,
    directory='models_dir',
    project_name='LSTM_EUR_regression_training'
)

try:
    # Train the model with hyperparameter tuning
    tuner.search(train_X, train_y, epochs=5, validation_data=(test_X, test_y))
    
    # Get the best model
    best_model = tuner.get_best_models(1)[0]
    
    # Evaluate the model
    evaluation = best_model.evaluate(test_X, test_y)
    print(f"Test Loss: {evaluation}")

    # Fit the network with the best model
    history = best_model.fit(
        train_X, train_y, 
        epochs=50, 
        batch_size=72, 
        validation_data=(test_X, test_y), 
        verbose=2, 
        shuffle=False
    )

    # Plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='val')
    pyplot.legend()
    pyplot.show()

except Exception as e:
    print(f"An error occurred during model training: {e}")

# Make a prediction
yhat = best_model.predict(test_X)

# Calculate RMSE
rmse = sqrt(mean_squared_error(y, yhat))
print(f'Test RMSE: {rmse:.3f}')
# %%
# Classification training (buy signal)
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv'
data = pd.read_csv(
    file_path, 
    header=0
)

data = data.tail(1000)

df = preprocess_data(data)
features = df.drop(columns=['buy_signal', 'sell_signal','Close_Position'])  # All features except target
target = df[['buy_signal']]  # Target is 'buy_signal'

# Ensure target is categorical (0 or 1)
target = target.astype(int)

# Frame the dataset for multi-step forecasting
n_in = 240  # Number of past observations (1 Month)
n_out = 1  # Binary classification for one step ahead (Buy or Not Buy)

supervised_features = series_to_supervised(features.values, n_in=n_in, n_out=n_out)
supervised_target = series_to_supervised(target.values, n_in=n_in, n_out=n_out)

# Extract input (X) and output (y)
n_features = features.shape[1]
n_obs = n_in * n_features

X = supervised_features.iloc[:, :n_obs].values
y = supervised_target.iloc[:, n_in * target.shape[1]:].values

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape X for Conv1D input (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], n_in, -1))
test_X = test_X.reshape((test_X.shape[0], n_in, -1))

print("Train (X) shape:", train_X.shape)
print("Test (X) shape:", test_X.shape)
print("Train (Y) shape:", train_y.shape)
print("Test (Y) shape:", test_y.shape)

tuner = kt.Hyperband(
    hypermodel=lambda hp:model_build.build_LSTM_model(hp),
    objective='accuracy',
    max_epochs=100,
    factor=3,
    hyperband_iterations=1,
    directory='models_dir',
    project_name='LSTM_EUR_Classification_Tuning'
)

try:
    # Train the model with hyperparameter tuning
    tuner.search(train_X, train_y, epochs=5, validation_data=(test_X, test_y))
    
    # Get the best model
    best_model = tuner.get_best_models(1)[0]
    
    # Evaluate the model
    evaluation = best_model.evaluate(test_X, test_y)
    print(f"Test Accuracy: {evaluation[1]} | Precision: {evaluation[2]} | Recall: {evaluation[3]}")

    # Fit the network with the best model
    history = best_model.fit(
        train_X, train_y, 
        epochs=50, 
        batch_size=72, 
        validation_data=(test_X, test_y), 
        verbose=2, 
        shuffle=False
    )

    # Plot training history
    pyplot.plot(history.history['accuracy'], label='train_accuracy')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.legend()
    pyplot.title("Model Accuracy Over Epochs")
    pyplot.show()

except Exception as e:
    print(f"An error occurred during model training: {e}")

yhat = best_model.predict(test_X)

# Convert probabilities to binary class (0 or 1)
yhat_class = (yhat > 0.5).astype(int)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(test_y, yhat_class))

#%%
# Classification training (sell signal)
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\trade_signals.csv'
data = pd.read_csv(
    file_path, 
    header=0
)

data = data.tail(1000)

df = preprocess_data(data)
features = df.drop(columns=['buy_signal', 'sell_signal','Close_Position'])  # All features except target
target = df[['sell_signal']]  # Target is 'buy_signal'

# Ensure target is categorical (0 or 1)
target = target.astype(int)

# Frame the dataset for multi-step forecasting
n_in = 240  # Number of past observations (1 Month)
n_out = 1  # Binary classification for one step ahead (Buy or Not Buy)

supervised_features = series_to_supervised(features.values, n_in=n_in, n_out=n_out)
supervised_target = series_to_supervised(target.values, n_in=n_in, n_out=n_out)

# Extract input (X) and output (y)
n_features = features.shape[1]
n_obs = n_in * n_features

X = supervised_features.iloc[:, :n_obs].values
y = supervised_target.iloc[:, n_in * target.shape[1]:].values

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape X for Conv1D input (samples, timesteps, features)
train_X = train_X.reshape((train_X.shape[0], n_in, -1))
test_X = test_X.reshape((test_X.shape[0], n_in, -1))

print("Train (X) shape:", train_X.shape)
print("Test (X) shape:", test_X.shape)
print("Train (Y) shape:", train_y.shape)
print("Test (Y) shape:", test_y.shape)

tuner = kt.Hyperband(
    hypermodel=lambda hp:model_build.build_LSTM_model(hp),
    objective='accuracy',
    max_epochs=100,
    factor=3,
    hyperband_iterations=1,
    directory='models_dir',
    project_name='LSTM_EUR_Classification_Tuning'
)

try:
    # Train the model with hyperparameter tuning
    tuner.search(train_X, train_y, epochs=5, validation_data=(test_X, test_y))
    
    # Get the best model
    best_model = tuner.get_best_models(1)[0]
    
    # Evaluate the model
    evaluation = best_model.evaluate(test_X, test_y)
    print(f"Test Accuracy: {evaluation[1]} | Precision: {evaluation[2]} | Recall: {evaluation[3]}")

    # Fit the network with the best model
    history = best_model.fit(
        train_X, train_y, 
        epochs=50, 
        batch_size=72, 
        validation_data=(test_X, test_y), 
        verbose=2, 
        shuffle=False
    )

    # Plot training history
    pyplot.plot(history.history['accuracy'], label='train_accuracy')
    pyplot.plot(history.history['val_accuracy'], label='val_accuracy')
    pyplot.legend()
    pyplot.title("Model Accuracy Over Epochs")
    pyplot.show()

except Exception as e:
    print(f"An error occurred during model training: {e}")

yhat = best_model.predict(test_X)

# Convert probabilities to binary class (0 or 1)
yhat_class = (yhat > 0.5).astype(int)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(test_y, yhat_class))