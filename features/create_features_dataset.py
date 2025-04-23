#%%
import os
import re
import sys
import pandas as pd
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from all_indicators import generate_all_indicators

def load_strategies(module_path):
    """Dynamically loads all functions from the given module path."""
    spec = importlib.util.spec_from_file_location("all_strategies", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["all_strategies"] = module
    spec.loader.exec_module(module)
    return {name: func for name, func in vars(module).items() if callable(func)}

# Load strategies
strategies_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\features\all_strategies.py"
strategies = load_strategies(strategies_file)

# Load data with correct data types
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_signals.csv'
# Load data and ensure correct data types
stock_df = pd.read_csv(file_path, dtype={'Date': str, 'Time': str})

# Convert Date & Time to Datetime
stock_df['Datetime'] = pd.to_datetime(stock_df['Date'] + ' ' + stock_df['Time'], format='%Y%m%d %H:%M:%S')

# Set Datetime as index and drop unnecessary columns
stock_df.set_index('Datetime', inplace=True)

# Debugging checks
print(stock_df.info())  # Ensure 'Close' is present and numeric
print(stock_df.head())  # Preview data
if 'Close' not in stock_df.columns:
    raise ValueError("Column 'Close' is missing from stock_df!")

indicators_df = generate_all_indicators(stock_df)

# Create a copy of indicators_df and add strategy signals to it
signals_df = indicators_df.copy()

# Apply each strategy function to stock_df and add non-duplicate columns to signals_df
for name, strategy_func in strategies.items():
    try:
        result = strategy_func(stock_df)
        if isinstance(result, pd.DataFrame) and not result.empty:
            # Add only columns not already in signals_df
            new_cols = [col for col in result.columns if col not in signals_df.columns]
            if new_cols:
                for col in new_cols:
                    try:
                        signals_df[col] = result[col].values
                    except ValueError:
                        # Handle length mismatch by aligning indices
                        signals_df[col] = result[col].reindex(signals_df.index)
    except Exception as e:
        print(f"Error applying {name}: {e}")

# Final check for any duplicate columns
signals_df = signals_df.loc[:, ~signals_df.columns.duplicated()]
signals_df
#%%
# Feature Dataset Analysis
# Print shape
print(f"Shape of DataFrame: {signals_df.shape}")
def find_duplicate_columns(df):
    duplicate_cols = {}
    cols = [col for col in df.columns if col.lower() not in ["datetime", "time"]]  # Exclude datetime and Time

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if df[cols[i]].equals(df[cols[j]]):
                duplicate_cols[cols[i]] = cols[j]
    
    return duplicate_cols

duplicate_cols = find_duplicate_columns(signals_df)

if duplicate_cols:
    print("Duplicate Columns Found:")
    for col1, col2 in duplicate_cols.items():
        print(f"- {col1} is duplicate of {col2}")
else:
    print("No duplicate columns found.")

# Identify constant columns (same value throughout or all NaN)
constant_cols = [col for col in signals_df.columns if signals_df[col].nunique(dropna=True) <= 1]
if constant_cols:
    print(f"Constant Columns: {constant_cols}")
else:
    print("No constant columns found.")

# Check for NaN values
missing_values = signals_df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    print("\nColumns with Missing Values:")
    print(missing_values)
else:
    print("No missing values found.")

# Check for duplicate rows
duplicate_rows = signals_df.duplicated().sum()
print(f"\nDuplicate Rows: {duplicate_rows}")

# Print column data types
print("\nColumn Data Types:")
print(signals_df.dtypes)

# Display basic statistics
print("\nBasic Statistics:")
print(signals_df.describe(include="all"))

#%%
# Optional: Drop Duplicate and constant columns
# Drop duplicate columns
signals_df = signals_df.loc[:, ~signals_df.T.duplicated()]

# Drop constant columns
constant_cols = [col for col in signals_df.columns if signals_df[col].nunique(dropna=True) <= 1]
signals_df = signals_df.drop(columns=constant_cols)

# Drop columns with excessive missing values (e.g., more than 50% NaN)
threshold = 0.5 * len(signals_df)
missing_values = signals_df.isnull().sum()
cols_to_drop = missing_values[missing_values > threshold].index
signals_df = signals_df.drop(columns=cols_to_drop)

# Print final shape after cleaning
print(f"New Shape after dropping unnecessary columns: {signals_df.shape}")


#%%
# Save as CSV
output_file = 'EURUSD_1min_sampled_features.csv'
signals_df.to_csv(output_file, index=False)
print(f"CSV file saved as {output_file}")
# %%