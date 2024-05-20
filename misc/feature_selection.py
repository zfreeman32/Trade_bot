# In[1]
import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
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
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import keras
from keras.layers import Dropout
from keras import backend
from Strategies import call_Strategies
import ta

# In[10]:
# fix random seed for reproducibility
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
true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

# Analyst Rating and Events

#%% 
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis = 1)
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
Y = data_encoded['signals_long'].values
Y2 = data_encoded['signals_short'].values

#Feature Importance from ExtraTrees and Random Forest
model = ExtraTreesRegressor()
model.fit(X1,Y2)
feature_importance = model.feature_importances_

et_indices = np.argsort(feature_importance)[::-1]

model = RandomForestRegressor()
model.fit(X1,Y2)
feature_importance_RF = model.feature_importances_

rf_indices=np.argsort(feature_importance_RF)[::-1]
#####Plotting of feature importances
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.bar(data_encoded.columns[et_indices], feature_importance[et_indices])
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Extra Trees Feature Importances for Short Signals')
plt.subplots_adjust(bottom=0.3, top=0.9)
#
plt.subplot(1,2,2)
plt.bar(data_encoded.columns[rf_indices], feature_importance_RF[rf_indices], color='green')
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Random Forest Feature Importances for Short Signals')
plt.subplots_adjust(bottom=0.3, top=0.9)
plt.show()

# Do it again for long signals
#Feature Importance from ExtraTrees and Random Forest
model = ExtraTreesRegressor()
model.fit(X1,Y)
feature_importance = model.feature_importances_

et_indices = np.argsort(feature_importance)[::-1]

model = RandomForestRegressor()
model.fit(X1,Y)
feature_importance_RF = model.feature_importances_

rf_indices=np.argsort(feature_importance_RF)[::-1]
#####Plotting of feature importances
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.bar(data_encoded.columns[et_indices], feature_importance[et_indices])
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Extra Trees Feature Importances for Long Signals')
plt.subplots_adjust(bottom=0.3, top=0.9)
#
plt.subplot(1,2,2)
plt.bar(data_encoded.columns[rf_indices], feature_importance_RF[rf_indices], color='green')
plt.xticks(rotation=90)
plt.ylabel('Feature Importances')
plt.title('Random Forest Feature Importances for Long Signals')
plt.subplots_adjust(bottom=0.3, top=0.9)
plt.show()


# ### Setting up the Neural Networks and defining all the Regression models for Gas Turbine Compressor Coefficient
# - In this step, the neural networks with and without dropout are defined with 1 input layer and 2 hidden layers with 2 additional dropout layers for the dropout case. Once each of the neural networks is defined, they are trained based upon an 80/20 split of the turbine data (80% training data, 20% test data).
# - The regression models are also defined in their base state (hyperparameters can be added if needed for improvement). They are also trained on the same 80/20 split of data.
# - The trained models are then asked to predict the first 50 values of test data and plotted on line graphs.
# 
# 

# In[12]:


# Split Data to Train and Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X1, Y, test_size=0.2, random_state=seed)

# Defining a compressor model with dropout
DNNmodel = keras.models.Sequential()
DNNmodel.add(keras.layers.Dense(100, activation='relu', input_dim=225))
DNNmodel.add(Dropout(0.1))
DNNmodel.add(keras.layers.Dense(50, activation='relu'))
DNNmodel.add(Dropout(0.1))
DNNmodel.add(keras.layers.Dense(25, activation='relu'))
DNNmodel.add(keras.layers.Dense(1, activation='relu'))
DNNmodel.compile(optimizer='adam', loss='mae',metrics=['mse'])
print('Training: Neural Network with Dropout Long Signals')
DNNmodel.fit(X_Train, Y_Train, epochs=30, batch_size=10, verbose=0)

#Defining NN without dropout for compressor decay
SNNmodel = keras.models.Sequential()
SNNmodel.add(keras.layers.Dense(100, activation='relu', input_dim=225))
SNNmodel.add(keras.layers.Dense(50, activation='relu'))
SNNmodel.add(keras.layers.Dense(25, activation='relu'))
SNNmodel.add(keras.layers.Dense(1, activation='relu'))
SNNmodel.compile(optimizer='adam', loss='mae',metrics=['mse'])
print('Training: Neural Network without Dropout Long Signals')
SNNmodel.fit(X_Train, Y_Train, epochs=30, batch_size=10, verbose=0)

#Defining many different Regression models for comparison purposes
models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
# models.append(('NNwDropout', DNNmodel()))
# models.append(('NNwoDropout', SNNmodel()))
#Fitting models and evaluating the results
results = []
names = []
scoring = []

plt.rcParams["font.size"] = "5.4"
plt.figure(figsize=(10,10))
i=1
for name, model in models:
    print('Training:', name, ' model' )
    model.fit(X_Train, Y_Train)
    predictions = model.predict(X_Test)
    plt.subplot(3,3,i)
    i=i+1
    plt.rcParams["font.size"] = "5.4"
    plt.plot(Y_Test[7000:7200], 'b-')
    plt.plot(predictions[7000:7200], 'r--')
    plt.title("%s %s" % (name,'Model Prediction for signals_long'))
    plt.ylabel('Prediction/Actual Signal')
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

predictions=DNNmodel.predict(X_Test)
name='NNwDropout'
plt.subplot(3,3,8)
plt.plot(Y_Test[0:49], 'b-')
plt.plot(predictions[0:49], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for signals_long'))
plt.ylabel('Prediction/Actual Signal')
plt.xlabel('Observation #')

score = explained_variance_score(Y_Test, predictions)
mae = mean_absolute_error(predictions, Y_Test)
results.append(mae)
scoring.append(score)
names.append(name)

predictions=SNNmodel.predict(X_Test)
name='NNwoDropout'
plt.subplot(3,3,9)
plt.plot(Y_Test[0:49], 'b-')
plt.plot(predictions[0:49], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for signals_long'))
plt.ylabel('Prediction/Actual Signal')
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


long_signals_results = {'Regression Model': names, 'Explained Variance Score': scoring, 'Mean Absolute Error': results}
long_signals_df = pd.DataFrame(long_signals_results, columns = ['Regression Model', 'Explained Variance Score', 'Mean Absolute Error'])
print(long_signals_df)
# ### Setting up new training and test data and retraining the algorithms for the Gas Turbine Decay coefficient

# In[14]:


# Split Data to Train and Test
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X1, Y2, test_size=0.2, random_state=seed)

############Same as above but for turbine decay state parameter
print('Training: Neural Network with Dropout')
DNNmodel.fit(X_Train, Y_Train, epochs=30, batch_size=10, verbose=0)
print('Training: Neural Network without Dropout')
SNNmodel.fit(X_Train, Y_Train, epochs=30, batch_size=10, verbose=0)

# Evaluations
results = []
names = []
scoring = []
plt.rcParams["font.size"] = "5.4"
plt.figure(figsize=(10,10))
i=1
for name, model in models:
    print('Training:', name, ' model' )
    model.fit(X_Train, Y_Train)
    predictions = model.predict(X_Test)
    plt.subplot(3,3,i)
    i=i+1
    plt.rcParams["font.size"] = "5.4"
    plt.plot(Y_Test[0:49], 'b-')
    plt.plot(predictions[0:49], 'r--')
    plt.title("%s %s" % (name,'Model Prediction for signals_short'))
    plt.ylabel('Prediction/Actual Signal')
    plt.xlabel('Observation #')
    # Evaluate the model
    score = explained_variance_score(Y_Test, predictions)
    mae = mean_absolute_error(predictions, Y_Test)
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mae)
    scoring.append(score)
    names.append(name)
predictions=DNNmodel.predict(X_Test)
name='NNwDropout'
plt.subplot(3,3,8)
plt.plot(Y_Test[0:49], 'b-')
plt.plot(predictions[0:49], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for signals_short'))
plt.ylabel('Prediction/Actual Signal')
plt.xlabel('Observation #')
score = explained_variance_score(Y_Test, predictions)
mae = mean_absolute_error(predictions, Y_Test)
results.append(mae)
scoring.append(score)
names.append(name)
predictions=SNNmodel.predict(X_Test)
name='NNwoDropout'
plt.subplot(3,3,9)
plt.plot(Y_Test[7000:7200], 'b-')
plt.plot(predictions[7000:7200], 'r--')
plt.title("%s %s" % (name, 'Model Prediction for signals_short'))
plt.ylabel('Prediction/Actul Signal')
plt.xlabel('Observation #')
plt.subplots_adjust(hspace=0.4, wspace=0.3)
score = explained_variance_score(Y_Test, predictions)
mae = mean_absolute_error(predictions, Y_Test)
results.append(mae)
scoring.append(score)
names.append(name)
plt.show()


# ### Printing the results of each algorithm in a table

# In[15]:


short_signals_results = {'Regression Model': names, 'Explained Variance Score': scoring, 'Mean Absolute Error': results}
short_signals_df = pd.DataFrame(short_signals_results, columns = ['Regression Model', 'Explained Variance Score', 'Mean Absolute Error'])
print(short_signals_df)