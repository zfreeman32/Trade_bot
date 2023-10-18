import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from prophet import Prophet
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from joblib import dump
from Strategies import call_Strategies
import ta
import numpy as np
from prophet.serialize import model_to_json

spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=spy_data.index)
# Add all technical indicators using TA library
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)

all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')

# True Signals (The most Optimal Buy/Sell Points since 1993)
true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

#%% 
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis = 1)
df = df.fillna(0)
df = df.replace('nan', 0)
df['Date'] = df.index.astype(float)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(df, columns=categorical_columns)


# In[]
X = data_encoded.iloc[:, :-2].values
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)
Y = data_encoded['signals_long'].values
Y2 = data_encoded['signals_short'].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print("Shapes - X_train:", X_train.shape, "X_test:", X_test.shape)
print("Shapes - y_train:", y_train.shape, "y_test:", y_test.shape)

# Train individual models
# Train the Random Forest model on the original data
X_train_rf = X_train.reshape(X_train.shape[0], -1)
X_test_rf = X_test.reshape(X_test.shape[0], -1)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_rf, y_train)

# Prophet
prophet_df = pd.DataFrame({'ds': spy_data['Date'], 'y': data_encoded['signals_long']})
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%m/%d/%Y')  # Use the correct date format
prophet_df.dropna(subset=['ds'], inplace=True)
prophet_model = Prophet()
prophet_model.fit(prophet_df)

# XGBoost
xgb_model = XGBClassifier()
xgb_model.fit(X_train_rf, y_train)

y_train_reshaped = np.reshape(y_train, (y_train.shape[0], 1, 1))

# LSTM
lstm_model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
lstm_model.fit(X_train.astype('float32'), y_train_reshaped.astype('float32'), epochs=10, batch_size=64)

# GRU
gru_model = Sequential([
    GRU(50, input_shape=(X_train.shape[1],X_train.shape[2]), activation='relu'),
    Dense(1, activation='sigmoid')
])
gru_model.compile(loss='binary_crossentropy', optimizer='adam')
gru_model.fit(X_train.astype('float32'), y_train_reshaped.astype('float32'), epochs=10, batch_size=64)

# Make predictions with each model
rf_predictions = rf_model.predict(X_test_rf)
xgb_predictions = xgb_model.predict(X_test_rf)
lstm_predictions = (lstm_model.predict(X_test.astype('float32')) > 0.5).astype(int)
gru_predictions = (gru_model.predict(X_test.astype('float32')) > 0.5).astype(int)

future = prophet_model.make_future_dataframe(periods=len(y_test))
prophet_forecast = prophet_model.predict(future)
prophet_predictions = prophet_forecast.tail(len(y_test))['yhat'].values

rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f'Random Forest Model Accuracy: {rf_accuracy:.2f}')

xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print(f'XGB Model Accuracy: {rf_accuracy:.2f}')

lstm_accuracy = accuracy_score(y_test, lstm_predictions)
print(f'LSTM Accuracy: {rf_accuracy:.2f}')

gru_accuracy = accuracy_score(y_test, gru_predictions)
print(f'GRU Model Accuracy: {rf_accuracy:.2f}')

# Save the trained models
dump(rf_model, 'rf_long_model.joblib')
dump(xgb_model, 'xgb_long_model.joblib')
lstm_model.save('lstm_long_model.h5')
gru_model.save('gru_long_model.h5')
with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(m)) 
# %%
