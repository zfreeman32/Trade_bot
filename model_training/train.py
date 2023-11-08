from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, ElasticNet, Lasso
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


import pandas as pd
from Strategies import call_Strategies
import ta
import numpy as np

seed = 7
np.random.seed(seed)

# In[11]:
spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
# Convert the data to a Pandas DataFrame
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=spy_data.index)
# Add all technical indicators using TA library
indicators_df = ta.add_all_ta_features(
    spy_data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
print(indicators_df.columns)

all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
print(all_signals_df)

# True Signals (The most Optimal Buy/Sell Points since 1993)
# true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

# Analyst Rating and Events

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
linear_reg = LinearRegression()
linear_reg.fit(X_train, Y_train)
linear_reg_predictions = linear_reg.predict(X_test)
mse_linear_reg = mean_squared_error(Y_test, linear_reg_predictions)
print("Linear Regression Mean Squared Error:", mse_linear_reg)

#In[]
# Ridge Regression model
ridge_reg = Ridge(alpha=1.0)  # You can adjust the alpha parameter
ridge_reg.fit(X_train, Y_train)
ridge_reg_predictions = ridge_reg.predict(X_test)
mse_ridge_reg = mean_squared_error(Y_test, ridge_reg_predictions)
print("Ridge Regression Mean Squared Error:", mse_ridge_reg)

#In[]
# Logistic Regression model
logistic_reg = LogisticRegression()
logistic_reg.fit(X_train, Y_train)
logistic_reg_predictions = logistic_reg.predict(X_test)
accuracy = accuracy_score(Y_test, logistic_reg_predictions)  # You'll need to import accuracy_score
print("Logistic Regression Accuracy:", accuracy)

# %%
# Decision Tree model
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, Y_train)
decision_tree_predictions = decision_tree.predict(X_test)
mse_decision_tree = mean_squared_error(Y_test, decision_tree_predictions)
print("Decision Tree Mean Squared Error:", mse_decision_tree)

#%%
# Random Forest model
random_forest = RandomForestRegressor(n_estimators=100)  # You can adjust the number of estimators
random_forest.fit(X_train, Y_train)
random_forest_predictions = random_forest.predict(X_test)
mse_random_forest = mean_squared_error(Y_test, random_forest_predictions)
print("Random Forest Mean Squared Error:", mse_random_forest)

#%%
# LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')  # You can use other loss functions depending on your task
model.fit(X_train, Y_train, epochs=10, batch_size=32)  # You can adjust the number of epochs and batch size
loss = model.evaluate(X_test, Y_test)
print("LSTM Test Loss:", loss)

#%%
# 1D CNN
model = Sequential()
model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')  # You can use other loss functions depending on your task
model.fit(X_train, Y_train, epochs=10, batch_size=32)  # You can adjust the number of epochs and batch size
loss = model.evaluate(X_test, Y_test)
print("CNN Test Loss:", loss)

#%%
# RNN
model = Sequential()
model.add(SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer
model.compile(optimizer='adam', loss='mean_squared_error')  # You can use other loss functions depending on your task
model.fit(X_train, Y_train, epochs=10, batch_size=32)  # You can adjust the number of epochs and batch size
loss = model.evaluate(X_test, Y_test)
print("RNN Test Loss:", loss)

#%%
# Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)
Y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("Naive Bayes Accuracy:", accuracy)

#%%
# Elastic Net
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic_net.fit(X_train, Y_train)
Y_pred = elastic_net.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Elastic Net Mean Squared Error:", mse)

#%%
# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, Y_train)
Y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print("SVM Accuracy:", accuracy)

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
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=10, batch_size=32)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("DNN Mean Squared Error:", mse)

#%%
# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"  # You can use other models like "gpt2-medium" for better performance
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
input_text = "Today's close price is $100. Tomorrow's close price will be "
input_ids = tokenizer.encode(input_text, return_tensors="pt")
with torch.no_grad():
    output = model.generate(input_ids, max_length=20, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50)
predicted_prices = [tokenizer.decode(output[0], skip_special_tokens=True)]
print("Predicted Close Prices:", predicted_prices)
