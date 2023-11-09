import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
# from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, ElasticNet, Lasso
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from keras.wrappers.scikit_learn import KerasRegressor
from prophet import Prophet
from prophet.serialize import model_to_json
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
import pandas as pd
from Strategies import call_Strategies
import ta
import numpy as np

seed = 4
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

# In[]
X = data_encoded.iloc[:, :-2].values
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)
Y = data_encoded['Close'].values
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=seed)

#In[]
# Linear Regression model
linear_reg = linear_model.LinearRegression()
linear_reg.fit(X_train, Y_train)
linear_reg_predictions = linear_reg.predict(X_test)
mse_linear_reg = mean_squared_error(Y_test, linear_reg_predictions)
print("Linear Regression Mean Squared Error:", mse_linear_reg)

#In[]
# Ridge Regression model
ridge_reg = linear_model.Ridge() 
param_grid = {'alpha': [0.1, 1.0, 10.0]}
grid_search = GridSearchCV(ridge_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_ # You can adjust the alpha parameter
ridge_reg_predictions = best_model.predict(X_test)
mse_ridge_reg = mean_squared_error(Y_test, ridge_reg_predictions)
print("Ridge Regression Mean Squared Error:", mse_ridge_reg)
print(best_params)
print(best_model)

#In[]
# Logistic Regression model
# logistic_reg = linear_model.LogisticRegression()
# logistic_reg.fit(X_train, Y_train)
# logistic_reg_predictions = logistic_reg.predict(X_test)
# accuracy = accuracy_score(Y_test, logistic_reg_predictions)  # You'll need to import accuracy_score
# print("Logistic Regression Accuracy:", accuracy)

# %%
# Decision Tree model
# Define a parameter grid to search
param_grid = {
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
decision_tree = DecisionTreeRegressor()
grid_search = GridSearchCV(decision_tree, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_decision_tree = grid_search.best_estimator_
decision_tree_predictions = best_decision_tree.predict(X_test)
mse_decision_tree = mean_squared_error(Y_test, decision_tree_predictions)
print("Best Decision Tree Mean Squared Error:", mse_decision_tree)
print(best_params)
print(best_decision_tree)

#%%
# Random Forest model
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # You can adjust the number of estimators
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_forest = RandomForestRegressor()  # You can adjust the number of estimators
grid_search = GridSearchCV(random_forest, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
best_random_forest = grid_search.best_estimator_
random_forest_predictions = best_random_forest.predict(X_test)
mse_random_forest = mean_squared_error(Y_test, random_forest_predictions)
print("Best Random Forest Mean Squared Error:", mse_random_forest)
print(best_params)
print(best_random_forest)

#%%
# LSTM

# Define a function to create your LSTM model
def create_lstm_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

lstm_model = KerasRegressor(build_fn=create_lstm_model, epochs=10, batch_size=32, verbose=0)
param_dist = {
    'units': [50, 100, 150],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}
random_search = RandomizedSearchCV(lstm_model, param_distributions=param_dist, cv=3, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, Y_train)
best_params = random_search.best_params_
best_lstm_model = random_search.best_estimator_
loss = best_lstm_model.model.evaluate(X_test, Y_test)
print("Best LSTM Test Loss:", loss)
print(best_params)
print(best_lstm_model)

#%%
# GBT
gbt = GradientBoostingClassifier()
params_gbt = {
    'n_estimators': [100, 200, 300],  # Number of boosting stages to be used
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
    'max_depth': [3, 4, 5]  # Maximum depth of individual trees
}
gbt_gs = GridSearchCV(gbt, params_gbt, cv=5)
gbt_gs.fit(X_train, Y_train)
best_gbt = gbt_gs.best_estimator_
print("Best Hyperparameters:", gbt_gs.best_params_)
predictions = best_gbt.predict(X_test)
loss = best_gbt.model.evaluate(X_test, Y_test)
print("Best LSTM Test Loss:", loss)

#%%
# 1D CNN
# Create a function to build the CNN model
def create_cnn_model(filters=32, kernel_size=3, optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
cnn_model = KerasRegressor(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=0)
cnn_param_dist = {
    'filters': [16, 32, 64],
    'kernel_size': [3, 5, 7],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}
random_search = RandomizedSearchCV(cnn_model, param_distributions=param_dist, cv=3, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, Y_train)
best_params = random_search.best_params_
best_cnn_model = random_search.best_estimator_
loss = best_cnn_model.model.evaluate(X_test, Y_test)
print("Best LSTM Test Loss:", loss)
print(best_params)
print(best_cnn_model)

#%%
# RNN
def create_rnn_model(units=50, optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
rnn_model = KerasRegressor(build_fn=create_cnn_model, epochs=10, batch_size=32, verbose=0)
rnn_param_dist = {
    'units': [50, 100, 150],
    'optimizer': ['adam', 'sgd', 'rmsprop']
}
random_search = RandomizedSearchCV(rnn_model, param_distributions=param_dist, cv=3, n_iter=10, scoring='neg_mean_squared_error', n_jobs=-1)
random_search.fit(X_train, Y_train)
best_params = random_search.best_params_
best_rnn_model = random_search.best_estimator_
loss = best_rnn_model.model.evaluate(X_test, Y_test)
print("Best LSTM Test Loss:", loss)
print(best_params)
print(best_rnn_model)

#%%
# Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)
Y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Naive Bayes Accuracy:", accuracy)

#%%
# Elastic Net
elastic_net = linear_model.ElasticNet()
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.5, 0.9],
}
grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, Y_train)
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']
optimal_elastic_net = linear_model.ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio)
optimal_elastic_net.fit(X_train, Y_train)
Y_pred = optimal_elastic_net.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Optimized Elastic Net Mean Squared Error:", mse)
print(best_alpha)
print(optimal_elastic_net)

#%%
# Lasso
lasso = linear_model.Lasso()
param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
}
grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, Y_train)
best_alpha = grid_search.best_params_['alpha']
optimal_lasso = linear_model.Lasso(alpha=best_alpha)
optimal_lasso.fit(X_train, Y_train)
Y_pred = optimal_lasso.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Optimized Lasso Mean Squared Error:", mse)
print(best_alpha)
print(optimal_lasso)

#%%
# SVM
svm = SVR()
param_grid = {
    'C': [1, 10, 100],
    'epsilon': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf'],
}
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, Y_train)
best_C = grid_search.best_params_['C']
best_epsilon = grid_search.best_params_['epsilon']
best_kernel = grid_search.best_params_['kernel']
optimal_svm = SVR(C=best_C, epsilon=best_epsilon, kernel=best_kernel)
optimal_svm.fit(X_train, Y_train)
Y_pred = optimal_svm.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Optimized SVM Mean Squared Error:", mse)
print(best_C)
print(optimal_svm)
#%%
# Deep Boltzmann Machines
pretrained_dbm = keras.applications.VGG16(weights='imagenet', include_top=False)
model = keras.Sequential()
model.add(pretrained_dbm)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=10, batch_size=32)
accuracy = model.evaluate(X_test, Y_test)
print("DBM Model Accuracy:", accuracy)

#%%
# deep neural network model
def create_dnn_model(optimizer='adam', neurons=128):
    model = keras.Sequential([
        keras.layers.Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
dnn_regressor = KerasRegressor(build_fn=create_dnn_model, verbose=0)
param_grid = {
    'neurons': [64, 128, 256],
    'batch_size': [16, 32, 64],
    'epochs': [10, 20, 30],
}
grid_search = GridSearchCV(estimator=dnn_regressor, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3)
grid_search.fit(X_train, Y_train)
best_neurons = grid_search.best_params_['neurons']
best_batch_size = grid_search.best_params_['batch_size']
best_epochs = grid_search.best_params_['epochs']
optimal_dnn_model = create_dnn_model(neurons=best_neurons)
optimal_dnn_model.fit(X_train, Y_train, batch_size=best_batch_size, epochs=best_epochs, verbose=0)
Y_pred = optimal_dnn_model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Optimized DNN Mean Squared Error:", mse)

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
# Load pre-trained GPT-2 model and tokenizer
# model_name = "gpt2"  # You can use other models like "gpt2-medium" for better performance
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# input_text = "Today's close price is $100. Tomorrow's close price will be "
# input_ids = tokenizer.encode(input_text, return_tensors="pt")
# with torch.no_grad():
#     output = model.generate(input_ids, max_length=20, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
# predicted_prices = [tokenizer.decode(output[0], skip_special_tokens=True)]
# print("Predicted Close Prices:", predicted_prices)
