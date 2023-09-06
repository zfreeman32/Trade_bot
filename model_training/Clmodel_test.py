#In[1]
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D

#In[2]
symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2021-12-31"

# Use yfinance to get the stock data
qqq_data = yf.download(symbol, start=start_date, end=end_date)

# Convert the data to a Pandas DataFrame
qqq_data = pd.DataFrame(qqq_data).reset_index(drop=True)
# calculate RSI
delta = qqq_data['Close'].diff()
gain = delta.mask(delta < 0, 0)
loss = -delta.mask(delta > 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))


Volume = qqq_data['Volume']
Close = qqq_data['Close']

# combine all the data into one dataframe
technical_indicators = pd.DataFrame({
    'RSI': rsi })

technical_indicators.dropna()
df = technical_indicators[20:]
df.round(3)
print(df)

#In[3]

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
