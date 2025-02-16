#%% Imports
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data import preprocess_data
import optuna
from dask.distributed import Client
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

seed = 7
np.random.seed(seed)

#%%
# Initialize Dask Client
client = Client(processes=False)

client

#%%
# Load Data Efficiently
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\trade_signals.csv'
data = pd.read_csv(file_path, header=0)

#%%
# Preprocessing
window_size = 120
df = preprocess_data.generate_features(data)
df[['buy_signal']] = df[['buy_signal']].astype(int).shift(-1).fillna(0)

#%%
# Convert to Float32 for Lower Memory Usage
df = df.astype(np.float32)

# Apply Sliding Window
features_array = sliding_window_view(df.drop(columns=['buy_signal', 'sell_signal', 'Close_Position']).values, window_size, axis=0)
target_array = df['buy_signal'][window_size:].values
#%%
# Train-Test Split (Lazy Evaluation)
features_array = features_array[:target_array.shape[0], :]

train_X, test_X, train_y, test_y = train_test_split(
    features_array, target_array, test_size=0.2, random_state=42, shuffle=False
)

#%%
# Standardize Data (No Full Memory Allocation)
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)  # Dask handles this lazily
test_X = scaler.transform(test_X)

# Compute Only When Needed (Process in Batches)
batch_size = 50000  # Adjust based on available memory
for i in range(0, train_X.shape[0], batch_size):
    batch_train_X = train_X[i:i+batch_size].compute()
    batch_train_y = train_y[i:i+batch_size].compute()

#%%
# Define Objective Function
def objective(trial):
    classifier_name = trial.suggest_categorical(
        "classifier", [
            "RandomForest", "GradientBoosting", "SVC", "AdaBoost",
            "Bagging", "ExtraTrees", "HistGradientBoosting", "MLP",
            "DecisionTree", "ExtraTree"
        ]
    )
    
    params = {}
    if classifier_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
        model = RandomForestClassifier(**params, random_state=seed)

    elif classifier_name == "GradientBoosting":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
        }
        model = GradientBoostingClassifier(**params, random_state=seed)

    elif classifier_name == "SVC":
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        model = SVC(**params, probability=True)

    elif classifier_name == "AdaBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        model = AdaBoostClassifier(**params, random_state=seed)

    elif classifier_name == "Bagging":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 100, step=10),
            "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        }
        model = BaggingClassifier(**params, random_state=seed)

    elif classifier_name == "ExtraTrees":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
        }
        model = ExtraTreesClassifier(**params, random_state=seed)

    elif classifier_name == "HistGradientBoosting":
        params = {
            "max_iter": trial.suggest_int("max_iter", 50, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        }
        model = HistGradientBoostingClassifier(**params, random_state=seed)

    elif classifier_name == "MLP":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(50,), (100,), (200,)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh", "logistic"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01, log=True),
        }
        model = MLPClassifier(**params, random_state=seed)

    elif classifier_name == "DecisionTree":
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
        }
        model = DecisionTreeClassifier(**params, random_state=seed)

    elif classifier_name == "ExtraTree":
        params = {
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
        }
        model = ExtraTreeClassifier(**params, random_state=seed)

    # Train Model
    model.fit(train_X_np, train_y_np)

    # Evaluate Model
    preds = model.predict(test_X_np)
    acc = accuracy_score(test_y_np, preds)

    # Store best model
    if classifier_name not in best_models or acc > best_models[classifier_name]["accuracy"]:
        best_models[classifier_name] = {
            "accuracy": acc,
            "params": params,
            "model": model
        }

    return acc

# Run Optuna Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# Save Detailed Report
with open("classification_report.txt", "w") as report_file:
    for model_name, model_data in best_models.items():
        preds = model_data["model"].predict(test_X_np)
        report_file.write(f"\n🏆 Best {model_name} Model 🏆\n")
        report_file.write(f"Best Parameters: {model_data['params']}\n")
        report_file.write(f"Accuracy: {model_data['accuracy']:.6f}\n")
        report_file.write(f"Classification Report:\n{classification_report(test_y_np, preds)}\n")
        report_file.write(f"Confusion Matrix:\n{confusion_matrix(test_y_np, preds)}\n")
        report_file.write("=" * 50 + "\n")

print("\n🚀 Results saved to classification_report.txt 🚀")

#%%