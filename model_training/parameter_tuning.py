from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
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
X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
y_train_reshaped = np.reshape(y_train, (y_train.shape[0], 1))

# Define a function that creates your LSTM/GRU model
def create_model(nodes=50, dense_units=1, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        LSTM(nodes, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), activation='relu'),
        Dense(dense_units, activation='sigmoid'),
        Dropout(dropout_rate)
    ])

    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate))
    return model

# Create a KerasClassifier
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=64)

# Define hyperparameter ranges
param_grid = {
    'nodes': [50, 100, 150],
    'dense_units': [1, 2, 3],
    'dropout_rate': [0.2, 0.4, 0.6],
    'learning_rate': [0.001, 0.01, 0.1]
}

# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=3)
grid_result = grid.fit(X_train_reshaped, y_train_reshaped)

# Print the best hyperparameters and their corresponding accuracy
print(f"Best: {grid_result.best_score_:.2f} using {grid_result.best_params_}")
