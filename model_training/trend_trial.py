#%%
import sys
sys.path.append(r'C:\Users\zebfr\Desktop\All Files\TRADING\Trading_Bot')
import pandas as pd
from sklearn.metrics import accuracy_score
from ta import add_all_ta_features
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaseEnsemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from Strategies import call_Strategies
import ta


seed = 7
np.random.seed(seed)
#%%

data = pd.read_csv('../Trading_Bot/data/SPY.csv')
# Convert the data to a Pandas DataFrame
data = pd.DataFrame(data).reset_index(drop=True)

# read in all features
indicators_df = pd.DataFrame(index=data.index)
# Add all technical indicators using TA library
indicators_df = ta.add_all_ta_features(
    data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
print(indicators_df.columns)

all_signals_df = call_Strategies.generate_all_signals('../Trading_Bot/data/SPY.csv', '../Trading_Bot/data/VIX.csv')
print(all_signals_df)

# True Signals (The most Optimal Buy/Sell Points since 1993)
true_signals_df = pd.read_csv("../Trading_Bot/data/SPY_true_signals.csv")

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

X = data_encoded.iloc[:, :-2].values
Y = data_encoded['signals_long'].values


scaler=MinMaxScaler()
scaler.fit(X)
X1=scaler.transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X1, Y, test_size=0.2, random_state=seed)
# %%
live_pred_data = data.iloc[-16:-11]

# %%
def _produce_prediction(data, window):
    data['pred'] = 0
    
    prediction = (data.shift(-window)['Close'] >= data['Close'])
    prediction = prediction.iloc[:-window]
    data['pred'] = prediction.astype(int)
    
    return data

data = _produce_prediction(data_encoded, window=15)
del (data['Close'])
data = data.dropna() # Some indicators produce NaN values for the first few rows, we just remove them here
data.tail()
# %%
def _train_random_forest(X_train, y_train, X_test, y_test):

    
    # Create a new random forest classifier
    rf = RandomForestClassifier()
    
    # Dictionary of all values we want to test for n_estimators
    params_rf = {'n_estimators': [110,130,140,150,160,180,200]}
    
    # Use gridsearch to test all values for n_estimators
    rf_gs = GridSearchCV(rf, params_rf, cv=5)
    
    # Fit model to training data
    rf_gs.fit(X_train, y_train)
    
    # Save best model
    rf_best = rf_gs.best_estimator_
    
    # Check best n_estimators value
    print(rf_gs.best_params_)
    
    prediction = rf_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return rf_best
    
rf_model = _train_random_forest(X_Train, Y_Train, X_Test, Y_Test)

#%%
def _train_KNN(X_train, y_train, X_test, y_test):

    knn = KNeighborsClassifier()
    # Create a dictionary of all values we want to test for n_neighbors
    params_knn = {'n_neighbors': np.arange(1, 25)}
    
    # Use gridsearch to test all values for n_neighbors
    knn_gs = GridSearchCV(knn, params_knn, cv=5)
    
    # Fit model to training data
    knn_gs.fit(X_train, y_train)
    
    # Save best model
    knn_best = knn_gs.best_estimator_
     
    # Check best n_neigbors value
    print(knn_gs.best_params_)
    
    prediction = knn_best.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return knn_best

knn_model = _train_KNN(X_Train, Y_Train, X_Test, Y_Test)


def train_GBT(X_train, y_train, X_test, y_test):
    # Create a GBT classifier
    gbt = GradientBoostingClassifier()
    
    # Define hyperparameters to search
    params_gbt = {
        'n_estimators': [100, 200, 300],  # Number of boosting stages to be used
        'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
        'max_depth': [3, 4, 5]  # Maximum depth of individual trees
    }
    
    # Use grid search to find the best hyperparameters
    gbt_gs = GridSearchCV(gbt, params_gbt, cv=5)
    
    # Fit the model to the training data
    gbt_gs.fit(X_train, y_train)
    
    # Get the best GBT model
    best_gbt = gbt_gs.best_estimator_
    
    # Print the best hyperparameters
    print("Best Hyperparameters:", gbt_gs.best_params_)
    
    # Make predictions on the test data
    predictions = best_gbt.predict(X_test)

    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
    
    return best_gbt

# Usage example:
gbt_model = train_GBT(X_Train, Y_Train, X_Test, Y_Test)

    
#%%
def _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test):
    
    # Create a dictionary of our models
    estimators=[('knn', knn_model), ('rf', rf_model), ('gbt', gbt_model)]
    
    # Create our voting classifier, inputting our models
    ensemble = VotingClassifier(estimators, voting='hard')
    
    #fit model to training data
    ensemble.fit(X_train, y_train)
    
    #test our model on the test data
    print(ensemble.score(X_test, y_test))
    
    prediction = ensemble.predict(X_test)

    print(classification_report(y_test, prediction))
    print(confusion_matrix(y_test, prediction))
    
    return ensemble

ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_Train, Y_Train, X_Test, Y_Test)


#%%
def cross_Validation(data):

    # Split data into equal partitions of size len_train
    
    num_train = 10 # Increment of how many starting points (len(data) / num_train  =  number of train-test sets)
    len_train = 40 # Length of each train-test set
    
    # Lists to store the results from each model
    rf_RESULTS = []
    knn_RESULTS = []
    ensemble_RESULTS = []
    
    i = 0
    while True:
        
        # Partition the data into chunks of size len_train every num_train days
        df = data_encoded.iloc[i * num_train : (i * num_train) + len_train]
        i += 1
        print(i * num_train, (i * num_train) + len_train)
        
        if len(df) < 40:
            break
        y = df['signals_long']
        features = [x for x in df.columns if x not in ['signals_long']]
        X = df[features]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 7 * len(X) // 10,shuffle=False)
        y_test = pd.DataFrame(y_test)

        rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
        knn_model = _train_KNN(X_train, y_train, X_test, y_test)
        ensemble_model = _ensemble_model(rf_model, knn_model, gbt_model, X_train, y_train, X_test, y_test)
        
        rf_prediction = rf_model.predict(X_test)
        knn_prediction = knn_model.predict(X_test)
        ensemble_prediction = ensemble_model.predict(X_test)
        
        print('rf prediction is ', rf_prediction)
        print('knn prediction is ', knn_prediction)
        print('ensemble prediction is ', ensemble_prediction)
        print('truth values are ', y_test.values)
        
        rf_accuracy = accuracy_score(y_test.values, rf_prediction)
        knn_accuracy = accuracy_score(y_test.values, knn_prediction)
        ensemble_accuracy = accuracy_score(y_test.values, ensemble_prediction)
        
        print(rf_accuracy, knn_accuracy, ensemble_accuracy)
        rf_RESULTS.append(rf_accuracy)
        knn_RESULTS.append(knn_accuracy)
        ensemble_RESULTS.append(ensemble_accuracy)
        
        
    print('RF Accuracy = ' + str( sum(rf_RESULTS) / len(rf_RESULTS)))
    print('KNN Accuracy = ' + str( sum(knn_RESULTS) / len(knn_RESULTS)))
    print('Ensemble Accuracy = ' + str( sum(ensemble_RESULTS) / len(ensemble_RESULTS)))
    
    
cross_Validation(data)

#%%
live_pred_data.head()

del(live_pred_data['close'])
prediction = ensemble_model.predict(live_pred_data)
print(prediction)