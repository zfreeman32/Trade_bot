import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier, SGDOneClassSVM, LogisticRegression
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import NearestCentroid, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from Strategies import call_Strategies
import ta

seed = 7
np.random.seed(seed)

#%%
# Read data
data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
data = pd.DataFrame(data).reset_index(drop=True)
indicators_df = pd.DataFrame(index=data.index)
indicators_df = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False)
all_signals_df = call_Strategies.generate_all_signals(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv', r'C:\Users\zeb.freeman\Documents\Trade_bot\data\VIX.csv')
true_signals_df = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY_true_signals.csv')

#%%
# Pre-process Data
df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis=1)
df = df.fillna(0)
df = df.replace('nan', 0)
df['Date'] = df.index.astype(float)
data_encoded = pd.get_dummies(df, columns=df.select_dtypes(include=['object', 'category']).columns)
X = data_encoded.iloc[:, :-2].values
Y = data_encoded['signals_long'].values
scaler = MinMaxScaler()
scaler.fit(X)
X1 = scaler.transform(X)

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X1, Y, test_size=0.2, random_state=seed)

classification_models = [
    (LogisticRegression(), {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}),
    (DecisionTreeClassifier(), {'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    (RandomForestClassifier(), {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30, 40]}),
    (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    (SVC(), {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly']}),
    (AdaBoostClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    (BaggingClassifier(), {'n_estimators': [10, 50, 100], 'max_samples': [1, 5, 10]}),
    (ExtraTreesClassifier(), {'n_estimators': [10, 50, 100]}),
    (GradientBoostingClassifier(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    (RandomForestClassifier(), {'n_estimators': [10, 50, 100]}),
    (HistGradientBoostingClassifier(), {'max_iter': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}),
    (GaussianProcessClassifier(), {}),
    (PassiveAggressiveClassifier(), {'C': [0.1, 1.0, 10.0], 'loss': ['hinge', 'squared_hinge']}),
    (Perceptron(), {'penalty': [None, 'l1', 'l2'], 'alpha': [0.0001, 0.001, 0.01]}),
    (RidgeClassifier(), {'alpha': [0.1, 1.0, 10.0]}),
    (SGDClassifier(), {'loss': ['hinge', 'log', 'modified_huber'], 'penalty': ['l1', 'l2', 'elasticnet']}),
    (SGDOneClassSVM(), {'nu': [0.01, 0.1, 0.5]}),
    (BernoulliNB(), {'alpha': [0.1, 1.0, 10.0]}),
    (CategoricalNB(), {'alpha': [0.1, 1.0, 10.0]}),
    (ComplementNB(), {'alpha': [0.1, 1.0, 10.0]}),
    (GaussianNB(), {}),
    (MultinomialNB(), {'alpha': [0.1, 1.0, 10.0]}),
    (NearestCentroid(), {'metric': ['euclidean', 'manhattan']}),
    (RadiusNeighborsClassifier(), {'radius': [0.5, 1.0, 1.5], 'outlier_label': [0, 1]}),
    (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}),
    (MLPClassifier(), {'hidden_layer_sizes': [(50,), (100,), (200,)], 'activation': ['relu', 'tanh', 'logistic']}),
    (SelfTrainingClassifier(), {}),
    (LinearSVC(), {'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge'], 'dual': [True, False], 'tol': [1e-3, 1e-4, 1e-5], 'C': [0.1, 1.0, 10.0]}),
    (NuSVC(), {'nu': [0.01, 0.1, 0.5], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}),
    (SVC(), {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']}),
    (DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    (ExtraTreeClassifier(), {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40]})
]

# Train and evaluate models
for model, param_grid in classification_models:
    if not param_grid:
        # If no parameters to tune, just fit the model
        model.fit(X_Train, Y_Train)
    else:
        # If there are parameters to tune, use GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_Train, Y_Train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_Test)

        # Evaluate model
        accuracy = accuracy_score(Y_Test, predictions)
        classification_rep = classification_report(Y_Test, predictions)

        print(f"{model.__class__.__name__} Metrics:\n"
              f"Accuracy: {accuracy}\n"
              f"Classification Report:\n{classification_rep}\n"
              f"Best Parameters: {best_params}\n"
              f"Best Model: {best_model}\n")