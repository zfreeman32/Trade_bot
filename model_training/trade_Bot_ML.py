# In[1]
import yfinance as yf
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import keras
from keras.layers import Dropout
import tensorflow as tf
from keras import backend

# Value > Avg and
# ADX[1] < ADX and
# RSI[1] < RSI and
# price[1] < trail[1] and
# price > trail

# In[10]:
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# In[11]:

symbol = "AAPL"
start_date = "2021-01-01"
end_date = "2021-12-31"

# Use yfinance to get the stock data
qqq_data = yf.download(symbol, start=start_date, end=end_date)

# Convert the data to a Pandas DataFrame
qqq_data = pd.DataFrame(qqq_data).reset_index(drop=True)

indicator = Indicators(qqq_data, start_date, end_date)


# %%

print(df)

# In[]
dataset = df.values
X = dataset[:, 0:13]
scaler = MinMaxScaler()
scaler.fit(X)
X1 = scaler.transform(X)
Y = dataset[:, 13]

# Feature Importance from ExtraTrees and Random Forest
model = ExtraTreesRegressor()
model.fit(X1, Y)
feature_importance = model.feature_importances_

et_indices = np.argsort(feature_importance)[::-1]

model = RandomForestRegressor()
model.fit(X1, Y)
feature_importance_RF = model.feature_importances_

rf_indices = np.argsort(feature_importance_RF)[::-1]
# Plotting of feature importances
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(df.columns[et_indices], feature_importance[et_indices])
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Extra Trees Feature Importances')
plt.subplots_adjust(bottom=0.3, top=0.9)
#
plt.subplot(1, 2, 2)
plt.bar(df.columns[rf_indices],
        feature_importance_RF[rf_indices], color='green')
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Random Forest Feature Importances')
plt.subplots_adjust(bottom=0.3, top=0.9)
plt.show()

# In[12]:
# Split Data to Train and Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X1, Y, test_size=0.2, random_state=seed)

# Defining a compressor model with dropout
DNNmodel = keras.models.Sequential()
DNNmodel.add(keras.layers.Dense(100, activation='relu', input_dim=13))
DNNmodel.add(Dropout(0.1))
DNNmodel.add(keras.layers.Dense(50, activation='relu'))
DNNmodel.add(Dropout(0.1))
DNNmodel.add(keras.layers.Dense(25, activation='relu'))
DNNmodel.add(keras.layers.Dense(1, activation='relu'))
DNNmodel.compile(optimizer='adam', loss='mae', metrics=['mse'])
print('Training: Neural Network with Dropout')
DNNmodel.fit(X_Train, Y_Train, epochs=X_Train.size,
             batch_size=X_Train.size, verbose='auto')

# Defining NN without dropout for compressor decay
SNNmodel = keras.models.Sequential()
SNNmodel.add(keras.layers.Dense(100, activation='relu', input_dim=13))
SNNmodel.add(keras.layers.Dense(50, activation='relu'))
SNNmodel.add(keras.layers.Dense(25, activation='relu'))
SNNmodel.add(keras.layers.Dense(1, activation='relu'))
SNNmodel.compile(optimizer='adam', loss='mae', metrics=['mse'])
print('Training: Neural Network without Dropout')
SNNmodel.fit(X_Train, Y_Train, epochs=X_Train.size,
             batch_size=X_Train.size, verbose="auto")

# In[]
# Defining many different Regression models for comparison purposes
models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('NNwDropout', DNNmodel))
models.append(('NNwoDropout', SNNmodel))
# Fitting models and evaluating the results
results = []
names = []
scoring = []

plt.rcParams["font.size"] = "5.4"
plt.figure(figsize=(10, 10))
i = 1
for name, model in models:
    print('Training:', name, ' model')
    model.fit(X_Train, Y_Train)
    predictions = model.predict(X_Test)
    plt.subplot(3, 3, i)
    i = i+1
    plt.rcParams["font.size"] = "5.4"
    plt.plot(Y_Test[0:49], 'b-')
    plt.plot(predictions[0:49], 'r--')
    plt.title("%s %s" % (name, 'Model Prediction for GT Compressor Decay'))
    plt.ylabel('Prediction/Actual Decay State')
    plt.xlabel('Observation #')
    # Evaluate the model
    score = explained_variance_score(Y_Test, predictions)
    mae = mean_absolute_error(predictions, Y_Test)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mae)
    scoring.append(score)
    names.append(name)

    msg = "%s: %f (%f)" % (name, score, mae)
    msg

predictions = DNNmodel.predict(X_Test)
name = 'NNwDropout'
plt.subplot(3, 3, 8)
plt.plot(Y_Test[0:49], 'b-')
plt.plot(predictions[0:49], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for GT Compressor Decay'))
plt.ylabel('Prediction/Actual Decay State')
plt.xlabel('Observation #')

score = explained_variance_score(Y_Test, predictions)
mae = mean_absolute_error(predictions, Y_Test)
results.append(mae)
scoring.append(score)
names.append(name)

predictions = SNNmodel.predict(X_Test)
name = 'NNwoDropout'
plt.subplot(3, 3, 9)
plt.plot(Y_Test[0:49], 'b-')
plt.plot(predictions[0:49], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for GT Compressor Decay'))
plt.ylabel('Prediction/Actual Decay State')
plt.xlabel('Observation #')
plt.subplots_adjust(hspace=0.4, wspace=0.3)
plt.show()
score = explained_variance_score(Y_Test, predictions)
mae = mean_absolute_error(predictions, Y_Test)
results.append(mae)
scoring.append(score)
names.append(name)


# ### Printing the results of each algorithm in a table

# In[13]:


gt_decay_results = {'Regression Model': names,
                    'Explained Variance Score': scoring, 'Mean Absolute Error': results}
gt_decay_df = pd.DataFrame(gt_decay_results, columns=[
                           'Regression Model', 'Explained Variance Score', 'Mean Absolute Error'])
gt_decay_df


# ### Setting up new training and test data and retraining the a

# In[15]:


gt_decay_results = {'Regression Model': names,
                    'Explained Variance Score': scoring, 'Mean Absolute Error': results}
gt_decay_df = pd.DataFrame(gt_decay_results, columns=[
                           'Regression Model', 'Explained Variance Score', 'Mean Absolute Error'])
gt_decay_df

# %%
