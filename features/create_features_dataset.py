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

def apply_strategy(strategy_func, stock_df):
    """Applies a strategy function to the stock DataFrame, ensuring consistent indexing."""
    try:
        result = strategy_func(stock_df)
        if isinstance(result, pd.DataFrame):
            result = result.reindex(stock_df.index)  # Enforce same index
            return result
    except Exception as e:
        print(f"Error applying {strategy_func.__name__}: {e}")
    return pd.DataFrame(index=stock_df.index)  # Return empty DataFrame if error

def generate_all_signals(stock_df, strategies):
    """Applies all loaded strategy functions using ThreadPoolExecutor, ensuring index consistency."""
    
    original_columns = stock_df.columns.tolist()  # Save original OHLCV columns

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda func: apply_strategy(func, stock_df), strategies.values()))

    # Ensure all DataFrames have the same index before concatenation
    valid_results = [df for df in results if not df.empty]

    # Convert all indices to match stock_df's index type
    for df in valid_results:
        df.index = stock_df.index

    # Concatenate stock_df with valid strategy outputs
    all_signals_df = pd.concat([df.filter(items=original_columns)] + valid_results, axis=1)
    all_signals_df = all_signals_df.loc[stock_df.index]  # Enforce exact index match

    return all_signals_df

# Load strategies
strategies_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\features\all_strategies.py"
strategies = load_strategies(strategies_file)

# Load data with correct data types
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_30min_sampled_signals.csv'
# Load data and ensure correct data types
stock_df = pd.read_csv(file_path, dtype={'Date': str, 'Time': str})

# Convert Date & Time to Datetime
stock_df['Datetime'] = pd.to_datetime(stock_df['Date'] + ' ' + stock_df['Time'], format='%Y%m%d %H:%M:%S')

# Set Datetime as index and drop unnecessary columns
stock_df.set_index('Datetime', inplace=True)
stock_df.drop(columns=['Date', 'Time'], inplace=True)

# Ensure numeric values
stock_df = stock_df.apply(pd.to_numeric, errors='coerce')

# Debugging checks
print(stock_df.info())  # Ensure 'Close' is present and numeric
print(stock_df.head())  # Preview data
if 'Close' not in stock_df.columns:
    raise ValueError("Column 'Close' is missing from stock_df!")

# Generate signals
signals_df = generate_all_signals(stock_df, strategies)
signals_df = generate_all_indicators(stock_df)

signals_df
#%%
# Feature Dataset Analysis
# Print shape
print(f"Shape of DataFrame: {signals_df.shape}")

# Identify duplicate columns
def find_duplicate_columns(df):
    duplicate_cols = {}
    cols = df.columns
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