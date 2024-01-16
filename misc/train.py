import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
# from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, ElasticNet, Lasso
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.metrics import MeanSquaredError, RootMeanSquaredError
from prophet import Prophet
from prophet.serialize import model_to_json
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
import pandas as pd
from Strategies import call_Strategies
import ta
import numpy as np

seed = 42
np.random.seed(seed)

# In[11]:
spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=spy_data.index)
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
# True Signals (The most Optimal Buy/Sell Points since 1993)
# true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

#%% 
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df], axis = 1)
df = df.fillna(0)
df = df.replace('nan', 0)
df['Date'] = df.index.astype(float)
categorical_columns = df.select_dtypes(include=['object', 'category']).columns

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(df, columns=categorical_columns)
print(data_encoded)
print(data_encoded.columns)

# %%
X = data_encoded.drop('Close', axis=1)
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)
Y = data_encoded['Close'].values
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=seed)

#%%
# Linear Regression model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, Y_train)
linear_reg_predictions = linear_reg.predict(X_test)
mse_linear_reg = mean_squared_error(Y_test, linear_reg_predictions)
evs_linear_reg = explained_variance_score(Y_test, linear_reg_predictions)
r2_linear_reg = r2_score(Y_test, linear_reg_predictions)
print(f"\nLinear Regression Stats:\n"
      f"Mean Squared Error: {mse_linear_reg}\n"
      f"Explained Variance Score: {evs_linear_reg}\n"
      f"R^2 Score: {r2_linear_reg}\n")

#%%
# Ridge Regression model
ridge_reg = linear_model.Ridge(max_iter=10000) 
param_grid = {'alpha': [i / 10.0 for i in range(1, 101)], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga'], 'tol': [.000001, .00001, .0001, .001, .01, .1]}
grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_ridge_params = grid_search.best_params_
best_ridge_model = grid_search.best_estimator_ # You can adjust the alpha parameter
ridge_reg_predictions = best_ridge_model.predict(X_test)
mse_ridge_reg = mean_squared_error(Y_test, ridge_reg_predictions)
evs_ridge_reg = explained_variance_score(Y_test, ridge_reg_predictions)
r2_ridge_reg = r2_score(Y_test, ridge_reg_predictions)
print(f"Ridge Regression Stats:\n"
      f"Mean Squared Error: {mse_ridge_reg}\n"
      f"Explained Variance Score: {evs_ridge_reg}\n"
      f"R^2 Score: {r2_ridge_reg}\n"
      f"Best Parameters: {best_ridge_params}\n"
      f"Best Ridge Regression Model: {best_ridge_model}\n")

#%%
# Lasso
lasso = linear_model.Lasso(max_iter=100000)
param_grid = {'alpha': [i / 10.0 for i in range(1, 101)], 'tol': [.000001, .00001, .0001, .001, .01, .1], 'selection': ['random']}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
optimal_lasso = grid_search.best_estimator_
optimal_lasso.fit(X_train, Y_train)
Y_pred = optimal_lasso.predict(X_test)
mse_lasso = mean_squared_error(Y_test, Y_pred)
r2_lasso = r2_score(Y_test, Y_pred)
print(f"Lasso Stats:\n"
      f"Mean Squared Error: {mse_lasso}\n"
      f"R^2 Score: {r2_lasso}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {optimal_lasso}\n")

#%%
# Decision Tree model
param_grid = {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'min_samples_split': [i for i in range(1, 11)], 'min_samples_leaf': [i for i in range(1, 11)]}
decision_tree = DecisionTreeRegressor()
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_decision_tree = grid_search.best_estimator_
decision_tree_predictions = best_decision_tree.predict(X_test)
mse_decision_tree = mean_squared_error(Y_test, decision_tree_predictions)
evs_decision_tree = explained_variance_score(Y_test, decision_tree_predictions)
r2_decision_tree = r2_score(Y_test, decision_tree_predictions)
print(f"Decision Tree Stats:\n"
      f"Mean Squared Error: {mse_decision_tree}\n"
      f"Explained Variance Score: {evs_decision_tree}\n"
      f"R^2 Score: {r2_decision_tree}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {best_decision_tree}\n")

#%%
# Random Forest model
param_grid = {'n_estimators': list(range(101)), 'min_samples_split': [i for i in range(1, 11)], 'min_samples_leaf': [i for i in range(1, 11)]}
random_forest = RandomForestRegressor()  # You can adjust the number of estimators
grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_random_forest = grid_search.best_estimator_
random_forest_predictions = best_random_forest.predict(X_test)
mse_random_forest = mean_squared_error(Y_test, random_forest_predictions)
evs_random_forest = explained_variance_score(Y_test, random_forest_predictions)
r2_random_forest = r2_score(Y_test, random_forest_predictions)
print(f"Random Forest Stats:\n"
      f"Mean Squared Error: {mse_random_forest}\n"
      f"Explained Variance Score: {evs_random_forest}\n"
      f"R^2 Score: {r2_random_forest}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {best_random_forest}\n")

#%%
# deep neural network model
def create_dnn_model(optimizer='adam', units=64):
    model = keras.Sequential([
        keras.layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

dnn_regressor = KerasRegressor(model=create_dnn_model, verbose=0, units=64)
param_grid = {
    'units': [4, 8, 16, 32, 64],  # Change 'neurons' to 'units'
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30],
}
grid_search = GridSearchCV(estimator=dnn_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, Y_train)
best_units = grid_search.best_params_['units']  # Update to 'units'
best_batch_size = grid_search.best_params_['batch_size']
best_epochs = grid_search.best_params_['epochs']
best_params = grid_search.best_params_
best_dnn_model = grid_search.best_estimator_
optimal_dnn_model = create_dnn_model(units=best_units)  # Update to 'units'
optimal_dnn_model.fit(X_train, Y_train, batch_size=best_batch_size, epochs=best_epochs, verbose=0)
Y_pred = optimal_dnn_model.predict(X_test)
mse_dnn = mean_squared_error(Y_test, Y_pred)
r2_dnn = r2_score(Y_test, Y_pred)
print(f"DNN Stats:\n"
      f"Mean Squared Error: {mse_dnn}\n"
      f"R^2 Score: {r2_dnn}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {best_dnn_model}\n")

#%%
# RNN
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
scaler_y = MinMaxScaler()
Y_train = scaler_y.fit_transform(Y_train.reshape(-1, 1)).reshape(-1)
Y_test = scaler_y.transform(Y_test.reshape(-1, 1)).reshape(-1)

def create_rnn_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

rnn_model = KerasRegressor(model=create_rnn_model, units = 50, epochs=10, batch_size=32, verbose=0)
rnn_param_dist = {
    'units': [50, 100, 150],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}
random_search = RandomizedSearchCV(rnn_model, param_distributions=rnn_param_dist, cv=3, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, Y_train)
best_params = random_search.best_params_
best_rnn_model = random_search.best_estimator_
y_pred = best_rnn_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
mse_rnn = mean_squared_error(Y_test, y_pred)
r2_rnn = r2_score(Y_test, y_pred)
print(f"RNN Stats:\n"
      f"Mean Squared Error: {mse_rnn}\n"
      f"R^2 Score: {r2_rnn}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {best_rnn_model}\n")

#%%
# LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
scaler_y = MinMaxScaler()
Y_train = scaler_y.fit_transform(Y_train.reshape(-1, 1)).reshape(-1)
Y_test = scaler_y.transform(Y_test.reshape(-1, 1)).reshape(-1)

def create_lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

lstm_model = KerasRegressor(model=create_lstm_model, epochs=10, batch_size=32, verbose=0)
param_dist = {
    'batch_size': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}
grid_search = GridSearchCV(lstm_model, param_dist, cv=3)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_lstm_model = grid_search.best_estimator_
y_pred = best_lstm_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
mse_lstm = mean_squared_error(Y_test, y_pred)
r2_lstm = r2_score(Y_test, y_pred)
print(f"LSTM Stats:\n"
      f"Mean Squared Error: {mse_lstm}\n"
      f"R^2 Score: {r2_lstm}\n"
      f"Best Parameters: {best_params}\n"
      f"Best Model: {best_lstm_model}\n")


# LightGBM
# XGBoost
# ADABoost
# MLP
# GRU

#%%
# Prophet
# prophet_df = pd.DataFrame({'ds': spy_data['Date'], 'y': data_encoded['signals_long']})
# prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%m/%d/%Y')  # Use the correct date format
# prophet_df.dropna(subset=['ds'], inplace=True)
# prophet_model = Prophet()
# prophet_model.fit(prophet_df)
# future = prophet_model.make_future_dataframe(periods=len(Y_test))
# prophet_forecast = prophet_model.predict(future)
# prophet_predictions = prophet_forecast.tail(len(Y_test))['yhat'].values

#%%
# Elastic Net
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
# param_grid = {
#     'l1_ratio': [0.1, 0.5, 0.7, 0.9],
# }
# elastic_net = linear_model.ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1], cv=5, random_state=0)
# grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
# grid_search.fit(X_train_scaled, Y_train)
# best_l1_ratio = grid_search.best_params_['l1_ratio']
# best_alpha = grid_search.best_params_['alphas']
# best_params = grid_search.best_params_
# optimal_elastic_net = linear_model.ElasticNetCV(l1_ratio=best_l1_ratio, alphas=best_alpha, cv=5, random_state=0)
# optimal_elastic_net.fit(X_train_scaled, Y_train)
# Y_pred = optimal_elastic_net.predict(X_test_scaled)
# mse_elastic = mean_squared_error(Y_test, Y_pred)
# r2_elastic = r2_score(Y_test, Y_pred)
# print(f"Elastic-Net Stats:\n"
#       f"Mean Squared Error: {mse_elastic}\n"
#       f"R^2 Score: {r2_elastic}\n"
#       f"Best Parameters: {best_params}\n"
#       f"Best Model: {optimal_elastic_net}\n\n")

# #%%
# # SVM
# scaler = MinMaxScaler()
# X_scaled = scaler.fit_transform(X)
# X_train_scaled, X_test_scaled, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=seed)
# print(data_encoded.shape)

# svm = SVR(max_iter=10000, epsilon= 0)
# param_grid = {
#     'C': [1, 10, 100],
#     'tol': [.000001, .00001, .0001, .001],
#     'degree': [1, 3, 5, 10, 100]
# }
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
# grid_search.fit(X_train_scaled, Y_train)
# best_C = grid_search.best_params_['C']
# best_epsilon = grid_search.best_params_['epsilon']
# best_kernel = grid_search.best_params_['kernel']
# best_params = grid_search.best_params_
# optimal_svm = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel)
# optimal_svm.fit(X_train_scaled, Y_train)
# Y_pred = optimal_svm.predict(X_test_scaled)
# mse_svm = mean_squared_error(Y_test, Y_pred)
# r2_svm = r2_score(Y_test, Y_pred)
# print(f"RNN Stats:\n"
#       f"Mean Squared Error: {mse_svm}\n"
#       f"R^2 Score: {r2_svm}\n"
#       f"Best Parameters: {best_params}\n"
#       f"Best Model: {optimal_svm}\n\n")