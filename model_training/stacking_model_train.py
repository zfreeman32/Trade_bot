#%% Regression Training Script for EUR/USD Forecasting
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import joblib
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
from data import preprocess_data

#%% Load Data
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\sampled_EURUSD_1min.csv'
data = pd.read_csv(file_path, header=0).tail(1000)

df = preprocess_data.generate_features(data)
features = df.drop(columns=['buy_signal', 'sell_signal', 'Close_Position'])
target = df[['buy_signal']].astype(int)

#%% Convert Series to Supervised Learning (Optional)
use_series_to_supervised = False  # Toggle this for experimentation

if use_series_to_supervised:
    n_in = 240  # Past observations (1 Month of 1-min bars)
    n_out = 15  # Binary classification (Buy/No Buy)

    supervised_features = preprocess_data.series_to_supervised(features.values, n_in=n_in, n_out=n_out)
    supervised_target = preprocess_data.series_to_supervised(target.values, n_in=n_in, n_out=n_out)

    # Extract inputs and outputs
    n_features = features.shape[1]
    n_obs = n_in * n_features

    X = supervised_features.iloc[:, :n_obs].values
    y = supervised_target.iloc[:, n_in * target.shape[1]:].values
else:
    X = features.values
    y = target.values

# Split dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Reshape X if using `series_to_supervised` for Conv1D compatibility
if use_series_to_supervised:
    train_X = train_X.reshape((train_X.shape[0], n_in, -1))
    test_X = test_X.reshape((test_X.shape[0], n_in, -1))

print("Train (X) shape:", train_X.shape)
print("Test (X) shape:", test_X.shape)
print("Train (Y) shape:", train_y.shape)
print("Test (Y) shape:", test_y.shape)

#%% Train a Stacking Regressor Using Multiple Models
# For this example, we will stack three regressors: RandomForest, XGBoost, and LightGBM.
def optimize_regressor(model_name):
    def objective_ind(trial):
        if model_name == "RandomForest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            }
            model = RandomForestRegressor(**params, random_state=42)
        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
            }
            model = XGBRegressor(**params, random_state=42)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
            }
            model = LGBMRegressor(**params, random_state=42)
        cv_scores = cross_val_score(model, train_X, train_y, cv=5, scoring="neg_mean_squared_error")
        cv_rmse = np.sqrt(-cv_scores).mean()
        return cv_rmse
    study_ind = optuna.create_study(direction="minimize", sampler=TPESampler())
    study_ind.optimize(objective_ind, n_trials=30, timeout=300)
    return study_ind.best_params

rf_best_params = optimize_regressor("RandomForest")
xgb_best_params = optimize_regressor("XGBoost")
lgb_best_params = optimize_regressor("LightGBM")

print("RF Best Params:", rf_best_params)
print("XGB Best Params:", xgb_best_params)
print("LGB Best Params:", lgb_best_params)

base_learners = [
    ('rf', RandomForestRegressor(**rf_best_params, random_state=42)),
    ('xgb', XGBRegressor(**xgb_best_params, random_state=42)),
    ('lgb', LGBMRegressor(**lgb_best_params, random_state=42))
]

meta_model = LinearRegression()

stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_model)
stacking_model.fit(train_X, train_y)

#%% Evaluate the Stacking Regressor
y_pred = stacking_model.predict(test_X)
mse = mean_squared_error(test_y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_y, y_pred)
r2 = r2_score(test_y, y_pred)
mape = np.mean(np.abs((test_y - y_pred) / test_y)) * 100

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

cv_rmse = np.sqrt(-cross_val_score(stacking_model, train_X, train_y, cv=5, scoring='neg_mean_squared_error'))
print(f"Cross-Validation RMSE Scores: {cv_rmse}")
print(f"Mean Cross-Validation RMSE: {cv_rmse.mean():.4f}")

#%% Compute Forecast Confidence Interval for New Data
# For a new data sample, we assume prediction error follows a normal distribution 
# with standard deviation equal to the mean cross-validated RMSE.
new_data = test_X[:5]  # Example: first 5 samples from the test set (replace with live data)
forecast = stacking_model.predict(new_data)
std_error = cv_rmse.mean()
ci_lower = forecast - 1.96 * std_error
ci_upper = forecast + 1.96 * std_error

print("Forecasted Close Prices:", forecast)
print("95% Confidence Interval Lower Bound:", ci_lower)
print("95% Confidence Interval Upper Bound:", ci_upper)

#%% Save the Stacking Model for Deployment
model_filename = "stacking_regressor_model.joblib"
joblib.dump(stacking_model, model_filename)
print(f"Stacking model saved to {model_filename}")

#%% Visualizations
# Residual Plot
residuals = test_y - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()

# Predicted vs Actual Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=test_y, y=y_pred, alpha=0.7)
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', lw=2)
plt.title("Predicted vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()