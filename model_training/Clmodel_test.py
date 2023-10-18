#In[1]
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
from ta import add_all_ta_features 
from sklearn.preprocessing import MinMaxScaler
from Strategies import call_Strategies
import ta

#In[2]
csv_file = r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv'
spy_data = pd.read_csv(csv_file)
# Convert the data to a Pandas DataFrame
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)

all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')

# True Signals as prediction column(The most Optimal Buy/Sell Points since 1993)
true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

#%% 
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis = 1)
df = df.fillna(0)
df = df.replace('nan', 0)
df['Date'] = df.index.astype(float)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to categorical columns
df = pd.get_dummies(df, columns=categorical_columns)
print(df)

#In[3]
df.drop(['Date'], axis=1, inplace=True)
# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


data = np.concatenate((train_data, test_data), axis=0)
X_train, y_train = df.iloc[:-1], pd.DataFrame(df['signals_short']) #signals_short
# y_train is (225,)
print(y_train)
print("Shapes - X_train:", X_train.shape, "y_train:", y_train.shape)

# Define the input and output data
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]


# Reshape the data for LSTM and GRU models
# Reshape the data for LSTM and GRU models
X_train = np.reshape(X_train, (X_train.shape[0], 224, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 224, 1))
print("Shapes - X_train:", X_train.shape, "X_test:", X_test.shape)

# Define the models
models = {
    'LSTM': Sequential([
        LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ]),
    'GRU': Sequential([
        GRU(50, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        Dense(1)
    ]),
    'CNN': Sequential([
        Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[0], X_train.shape[1])),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ]),
    'GBM': GradientBoostingClassifier(),
    'RF': RandomForestClassifier(),
    'SVM': SVC(),
    'LogReg': LogisticRegression()
}
#In[4]
# Create an empty dictionary to store model accuracies
model_accuracies = {}

# Loop through each model and calculate accuracy
for model_name, model in models.items():
    if isinstance(model, Sequential):  # For Keras models
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=0)
        y_pred = model.predict(X_test)
        y_pred = (y_pred > 0.5)  # Convert probabilities to binary predictions
        accuracy = accuracy_score(y_test, y_pred)
    else:  # For scikit-learn models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    model_accuracies[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy:.2f}')

# Print the accuracies for all models
print("Model Accuracies:")
for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy:.2f}")
