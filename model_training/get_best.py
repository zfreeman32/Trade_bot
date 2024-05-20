import os
import json

project_path = 'untitled_project'

def get_best_model():
    best_model = None
    best_val_loss = float('inf')

    for root, dirs, files in os.walk(project_path):
        if 'trial.json' in files:
            trial_path = os.path.join(root, 'trial.json')
            with open(trial_path, 'r') as f:
                trial_data = json.load(f)
                val_loss = trial_data.get('metrics', {}).get('val_loss', float('inf'))
                print(f"Checking {root}: val_loss = {val_loss}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = root

    return best_model

best_model_path = get_best_model()

if best_model_path:
    print(f"The best model is in the folder: {best_model_path}")
else:
    print("No models found in the specified directory.")
