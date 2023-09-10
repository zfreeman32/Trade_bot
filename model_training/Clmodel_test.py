#In[1]
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D
from ta import add_all_ta_features
from call_Strategies import generate_all_signals

#In[2]
csv_file = '../Trading_Bot/SPY.csv'
spy_data = pd.read_csv(csv_file)
# Convert the data to a Pandas DataFrame
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=spy_data.index)
# Add all technical indicators using TA library
indicators_df = add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
print(indicators_df.columns)

all_signals_df = generate_all_signals('SPY.csv', 'VIX.csv')
print(all_signals_df)

# True Signals (The most Optimal Buy/Sell Points since 1993)
true_signals_df = pd.read_csv("./true_signals/SPY_true_signals.csv")

# Analyst Rating and Events

#%% 
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis = 1)

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
X_train, y_train = data[:-1], data[-1]

# Define the input and output data
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Reshape the data for LSTM and GRU models
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

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
# Train and evaluate each model
for name, model in models.items():
    if name in ['LSTM', 'GRU', 'CNN']:
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X_train, y_train, epochs=50, batch_size=72, validation_data=(X_test, y_test), verbose=0)
        y_pred = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten() # Reshape y_test to match y_pred shape
        accuracy = accuracy_score(y_test, y_pred.flatten().round())
        print(f'{name} accuracy: {accuracy}')
    else:
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f'{name} accuracy: {accuracy}')

# %%
