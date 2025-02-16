import warnings
warnings.filterwarnings("ignore")
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from dask.distributed import Client
import dask.array as da
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# Initialize Dask Client
client = Client(processes=False)
seed = 42

# Load Data
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled_EURUSD_1min.csv'
data = pd.read_csv(file_path, header=0)

# Feature Engineering: Use last 240 time steps to predict next 15
window_size = 240
forecast_horizon = 15  # Predict next 15 steps

# Convert 'Close' column into a supervised learning format
X, y = [], []
for i in range(len(data) - window_size - forecast_horizon):
    X.append(data.iloc[i:i+window_size]['Close'].values)
    y.append(data.iloc[i+window_size:i+window_size+forecast_horizon]['Close'].values)

# Convert to Dask Array
X = da.from_array(np.array(X), chunks=(1000, window_size))
y = da.from_array(np.array(y), chunks=(1000, forecast_horizon))

# Split dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=False)

# Standardize Data
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Define Optuna Optimization Function
best_models = {}

train_X_np, train_y_np = train_X.compute(), train_y.compute()
test_X_np, test_y_np = test_X.compute(), test_y.compute()

def objective_reg(trial):
    """Objective function for Optuna optimization"""
    regressor_name = trial.suggest_categorical(
        "regressor", ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM",
                      "SVR", "MLP", "AdaBoost", "Bagging", "HistGradientBoosting",
                      "KNeighbors", "SGD"]
    )

    params = {}

    if regressor_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5)
        }
        model = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)

    elif regressor_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        model = GradientBoostingRegressor(**params, random_state=seed)

    elif regressor_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        model = XGBRegressor(**params, random_state=seed, n_jobs=-1)

    elif regressor_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        }
        model = LGBMRegressor(**params, random_state=seed, n_jobs=-1)

    elif regressor_name == "SVR":
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0, log=True)
        }
        model = SVR(**params)

    elif regressor_name == "MLP":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (200,)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        }
        model = MLPRegressor(**params, random_state=seed)

    elif regressor_name == "SGD":
        params = {
            "loss": trial.suggest_categorical("loss", ["squared_error", "huber", "epsilon_insensitive"])
        }
        model = SGDRegressor(**params, random_state=seed)

    elif regressor_name == "AdaBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50)
        }
        model = AdaBoostRegressor(**params, random_state=seed)

    elif regressor_name == "Bagging":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 30, step=10)
        }
        model = BaggingRegressor(**params, random_state=seed)

    elif regressor_name == "HistGradientBoosting":
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 200, step=50)
        }
        model = HistGradientBoostingRegressor(**params, random_state=seed)

    elif regressor_name == "KNeighbors":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 7, step=2)
        }
        model = KNeighborsRegressor(**params)

    # Train Model
    model.fit(train_X_np, train_y_np)

    # Predict & Evaluate
    preds = model.predict(test_X_np)
    rmse = mean_squared_error(test_y_np, preds, squared=False)

    # Store best model
    if regressor_name not in best_models or rmse < best_models[regressor_name]["rmse"]:
        best_models[regressor_name] = {
            "rmse": rmse,
            "params": params,
            "model": model
        }

    return rmse

# Run Optuna Optimization
study_reg = optuna.create_study(direction="minimize", sampler=TPESampler())
study_reg.optimize(objective_reg, n_trials=50, timeout=600)

# Save Results
with open("multi_step_forecasting_report.txt", "w") as report_file:
    for model_name, model_data in best_models.items():
        report_file.write(f"\n🏆 Best {model_name} Model 🏆\n")
        report_file.write(f"Best Parameters: {model_data['params']}\n")
        report_file.write(f"Best RMSE: {model_data['rmse']:.6f}\n")
        report_file.write("=" * 50 + "\n")

print("\n🚀 Results saved to multi_step_forecasting_report.txt 🚀")
