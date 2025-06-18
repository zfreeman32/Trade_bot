#%%
import pandas as pd
import os

# Mapping of signal type to feature file path
feature_files = {
    'short_signal': r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\feature_analysis_results_0523\short_signal_important_features.txt",
    'long_signal': r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\feature_analysis_results_0523\long_signal_important_features.txt",
    'Close': r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\feature_analysis_results_0523\Close_important_features.txt"
}

# Path to the large input CSV
input_csv = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\loki\workspace\EURUSD_1min_sampled_features.csv"  # <-- Change this

# Directory where outputs should go
output_dir = os.path.dirname(input_csv)

# Essential columns to always include
essential_columns = [
    'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'datetime', 'long_signal', 'short_signal', 'close_position'
]

# Helper to deduplicate while preserving order
def dedupe_preserve_order(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

# Function to extract unique features from a file
def read_unique_features(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    seen, features, duplicates = set(), [], []
    for line in lines:
        feature = line.strip()
        if not feature or feature.startswith('#') or feature.startswith('==='):
            continue
        if feature not in seen:
            seen.add(feature)
            features.append(feature)
        else:
            duplicates.append(feature)
    return features, duplicates

# Read the full header from the CSV once
full_header = pd.read_csv(input_csv, nrows=0).columns.tolist()

#%%
# Process each signal type
for signal_name, path in feature_files.items():
    print(f"\nProcessing {signal_name} features...")

    features, duplicates = read_unique_features(path)
    if duplicates:
        print(f"  Found duplicates (ignored): {duplicates}")

    # Build list of columns to keep for this signal
    columns_to_keep = dedupe_preserve_order(features + essential_columns)
    available_columns = [col for col in columns_to_keep if col in full_header]

    # Setup output path
    output_csv = os.path.join(output_dir, f"{signal_name}_dataset.csv")

    # Process in chunks to filter and save
    chunks = pd.read_csv(input_csv, chunksize=100000)
    filtered_chunks = [chunk[available_columns] for chunk in chunks]

    # Save result
    pd.concat(filtered_chunks).to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}")

# %%
