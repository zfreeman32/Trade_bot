import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
# from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, ElasticNet, Lasso
# from sklearn import linear_model
# from sklearn.svm import SVR
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import GradientBoostingClassifier
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, SimpleRNN
# from keras.wrappers.scikit_learn import KerasRegressor
# from prophet import Prophet
# from prophet.serialize import model_to_json
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch
import pandas as pd
from Strategies import call_Strategies
import ta
import numpy as np

seed = 4
np.random.seed(seed)

# In[11]:
eur_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\currency_data\eurusd_hour.csv')
eur_data = pd.DataFrame(eur_data).reset_index(drop=True)

#%%
# read in all features
indicators_df = pd.DataFrame(index=eur_data.index)
indicators_df = ta.add_all_ta_features(
    eur_data, open="Open", high="High", low="Low", close="Close", fillna=False
)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\currency_data\eurusd_hour.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
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