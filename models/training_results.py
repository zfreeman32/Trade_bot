#%%
import os
import json
import pandas as pd
from glob import glob

# Constants
ROOT_DIR = '.'  # or wherever your sell_trials_* directories are located

# Process each model folder separately
for model_dir in glob(os.path.join(ROOT_DIR, 'buy_trials_*')):
    model_name = os.path.basename(model_dir)
    trials_data = []

    for trial_dir in glob(os.path.join(model_dir, 'trial_*')):
        checkpoint_path = os.path.join(trial_dir, 'checkpoint.weights.h5')
        trial_json_path = os.path.join(trial_dir, 'trial.json')

        if not os.path.exists(trial_json_path):
            print(f"⛔ Missing trial.json in: {trial_dir}")
            continue
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Skipping {trial_dir} — No checkpoint.weights.h5")
            continue

        print(f"✅ Using {trial_dir}")
        with open(trial_json_path, 'r') as f:
            trial_data = json.load(f)

        trial_id = trial_data.get('trial_id')
        metrics = trial_data.get('metrics', {}).get('metrics', {})
        val_loss = metrics.get('val_loss', {}).get('observations', [{}])[0].get('value', [None])[0]
        val_accuracy = metrics.get('val_accuracy', {}).get('observations', [{}])[0].get('value', [None])[0]
        accuracy = metrics.get('accuracy', {}).get('observations', [{}])[0].get('value', [None])[0]

        row = {
            'model_type': model_name,
            'trial_id': trial_id,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'accuracy': accuracy,
        }

        hparams = trial_data.get('hyperparameters', {}).get('values', {})
        row.update(hparams)
        trials_data.append(row)

    # Save model-specific results if any found
    if trials_data:
        df_model = pd.DataFrame(trials_data)
        csv_output = f"{model_name}_successful_trials.csv"
        df_model.to_csv(csv_output, index=False)
        print(f"✅ CSV saved: {csv_output}")

        # Sort and take top 5 by val_loss
        top5_df = df_model.sort_values(by='val_loss', ascending=True).head(5)

        # Generate analysis
        analysis_lines = [
            f"Top 5 Trials for {model_name} by Validation Loss:\n",
            top5_df.to_string(index=False),
            "\n\nMost Common Hyperparameters in Top 5:\n"
        ]

        common_params = {}
        for col in top5_df.columns:
            if col not in ['trial_id', 'val_loss', 'val_accuracy', 'accuracy', 'model_type']:
                counts = top5_df[col].value_counts(dropna=True)
                if not counts.empty and counts.iloc[0] > 1:
                    common_params[col] = (counts.index[0], counts.iloc[0])

        if common_params:
            for k, (val, count) in common_params.items():
                analysis_lines.append(f"  {k}: {val} (shared by {count} models)")
        else:
            analysis_lines.append("  No shared hyperparameters among top 5.")

        analysis_output = f"{model_name}_top_5_models_analysis.txt"
        with open(analysis_output, 'w') as f:
            f.write('\n'.join(analysis_lines))
        print(f"📊 Analysis saved: {analysis_output}")
#%%