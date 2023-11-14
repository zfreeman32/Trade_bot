import sys
sys.path.append(r'C:\Users\zeb.freeman\Documents\Trade_bot')
# from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, ElasticNet, Lasso
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.linear_model import (
    SGDRegressor,
    Lars,
    LassoLars,
    OrthogonalMatchingPursuit,
    ARDRegression,
    BayesianRidge,
    SGDOneClassSVM,
    PoissonRegressor,
    TweedieRegressor,
    GammaRegressor,
    HuberRegressor,
    QuantileRegressor,
    RANSACRegressor,
    TheilSenRegressor,
    MultiTaskLasso,
    MultiTaskElasticNet,
    PassiveAggressiveRegressor
)
from sklearn.ensemble import IsolationForest
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from Strategies import call_Strategies
import ta
import numpy as np
seed = 42
np.random.seed(seed)

#%%
spy_data = pd.read_csv(r'C:\Users\zeb.freeman\Documents\Trade_bot\data\SPY.csv')
spy_data = pd.DataFrame(spy_data).reset_index(drop=True)
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
data_encoded = pd.get_dummies(df, columns=categorical_columns)

# %%
X = data_encoded.drop('Close', axis=1)
scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)
Y = data_encoded['Close'].values
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y, test_size=0.2, random_state=seed)
min_components = min(X_train.shape[0], X_train.shape[1], 1)


models = [
    (linear_model.LinearRegression(), {}),
    (linear_model.Ridge(max_iter=10000), {'alpha': [0.1, 1.0, 10.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'saga']}),
    (linear_model.Lasso(max_iter=100000), {'alpha': [0.01, .05, 0.1, 0.5, 1.0, 5.0, 10.0], 'tol': [.000001, .00001, .0001, .001, .01, .1]}),
    (DecisionTreeRegressor(), {'max_depth': [None, 10, 20, 30, 40], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    (RandomForestRegressor(), {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30, 40]}),
    (CCA(), {'n_components': [1, min_components]}),
    (PLSCanonical(), {'n_components': [1, min_components]}),
    (PLSRegression(), {'n_components': [1, min_components]}),
    (DummyRegressor(), {'strategy': ['mean', 'median', 'quantile', 'constant'], 'constant': [None], 'quantile': [None]}),
    (AdaBoostRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0]}),
    (BaggingRegressor(), {'n_estimators': [10, 20, 30], 'max_samples': [0.5, 0.8, 1.0], 'max_features': [0.5, 0.8, 1.0]}),
    (ExtraTreesRegressor(), {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}),
    (GradientBoostingRegressor(), {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [3, 4, 5]}),
    (HistGradientBoostingRegressor(), {'max_iter': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1.0], 'max_depth': [None, 10, 20, 30]}),
    (IsolationForest(), {'n_estimators': [50, 100, 200], 'max_samples': ['auto', 0.5, 0.8], 'contamination': ['auto', 0.1, 0.2]}),
    (GaussianProcessRegressor(), {}),
    (IsotonicRegression(), {}),
    (KernelRidge(), {'alpha': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}),
    (SGDRegressor(), {'loss': ['squared_loss', 'huber', 'epsilon_insensitive'], 'penalty': ['l2', 'l1', 'elasticnet']}),
    (Lars(), {'fit_intercept': [True, False]}),
    (LassoLars(), {'alpha': [0.1, 1.0, 10.0]}),
    (OrthogonalMatchingPursuit(), {}),
    (ARDRegression(), {'max_iter': [100, 200, 300]}),
    (BayesianRidge(), {'max_iter': [100, 200, 300]}),
    (SGDOneClassSVM(), {'nu': [0.1, 0.5, 0.9]}),
    (PoissonRegressor(), {'alpha': [0.1, 1.0, 10.0]}),
    (TweedieRegressor(), {'power': [0.5, 1.0, 1.5]}),
    (GammaRegressor(), {'alpha': [0.1, 1.0, 10.0]}),
    (HuberRegressor(), {'epsilon': [1.1, 1.2, 1.3]}),
    (QuantileRegressor(), {'alpha': [0.1, 0.5, 0.9]}),
    (RANSACRegressor(), {}),
    (TheilSenRegressor(), {}),
    (MultiTaskLasso(), {'alpha': [0.1, 1.0, 10.0]}),
    (MultiTaskElasticNet(), {'alpha': [0.1, 1.0, 10.0]}),
    (PassiveAggressiveRegressor(), {'C': [0.1, 1.0, 10.0]}),
    (RadiusNeighborsRegressor(), {'radius': [1.0, 2.0, 3.0]}),
    (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    (MLPRegressor(), {'hidden_layer_sizes': [(50,), (100,), (200,)], 'activation': ['relu', 'tanh', 'logistic']}),
    (LinearSVR(), {'epsilon': [0.1, 0.2, 0.3], 'C': [0.1, 1.0, 10.0]}),
    (NuSVR(), {'nu': [0.1, 0.5, 0.9], 'C': [0.1, 1.0, 10.0]}),
    (SVR(), {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1.0, 10.0]}),
    (ExtraTreeRegressor(), {'criterion': ['mse', 'mae']})
]

for model, param_grid in models:
    if not param_grid:
        # If no parameters to tune, just fit the model
        model.fit(X_train, Y_train)
    else:
        # If there are parameters to tune, use GridSearchCV
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, Y_train)
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        mse = mean_squared_error(Y_test, predictions)
        evs = explained_variance_score(Y_test, predictions)
        r2 = r2_score(Y_test, predictions)

        print(f"{model.__class__.__name__} Stats:\n"
              f"Mean Squared Error: {mse}\n"
              f"Explained Variance Score: {evs}\n"
              f"R^2 Score: {r2}\n"
              f"Best Parameters: {best_params}\n"
              f"Best Model: {best_model}\n")