#%%
import os
import re
import sys
import pandas as pd
import importlib.util
from concurrent.futures import ThreadPoolExecutor

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

#%%
# Load strategies
strategies_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\features\TOS_Strategies\all_strategies.py"
strategies = load_strategies(strategies_file)

# Load data with correct data types
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, dtype={'Date': str, 'Time': str})  # Convert Date & Time to strings

# Convert Date & Time to a single Datetime index
stock_df['Datetime'] = pd.to_datetime(stock_df['Date'] + ' ' + stock_df['Time'], format='%Y%m%d %H:%M:%S')

# Set Datetime as the index and drop original Date & Time columns
stock_df.set_index('Datetime', inplace=True)
print(stock_df.info())  # Ensure the index is DatetimeIndex
print(stock_df.head()) 
#%%
# Generate signals
signals_df = generate_all_signals(stock_df, strategies)

# Reset index if necessary
signals_df = signals_df.reset_index()

# Display output
signals_df.head()

# %%
# # %%
# import os
# import importlib.util
# import inspect
# import pandas as pd

# def load_functions_from_directory(directory):
#     functions = {}
#     for filename in os.listdir(directory):
#         if filename.endswith(".py") and filename != "__init__.py":
#             module_name = filename[:-3]  # Remove ".py"
#             module_path = os.path.join(directory, filename)
            
#             spec = importlib.util.spec_from_file_location(module_name, module_path)
#             module = importlib.util.module_from_spec(spec)
#             spec.loader.exec_module(module)

#             # Extract functions from the module
#             for name, func in inspect.getmembers(module, inspect.isfunction):
#                 functions[name] = func  # Store function reference
#     return functions

# def run_all_functions(functions, data):
#     results_dict = {}  # Dictionary to store outputs

#     for name, func in functions.items():
#         try:
#             output = func(data)  # Call each function
            
#             # Convert output into a DataFrame column
#             if isinstance(output, pd.DataFrame):
#                 results_dict[name] = output.iloc[:, 0]  # Take the first column if it's a DataFrame
#             elif isinstance(output, pd.Series):
#                 results_dict[name] = output
#             elif isinstance(output, (list, tuple)):
#                 results_dict[name] = pd.Series(output)  # Convert list to Series
#             else:
#                 results_dict[name] = pd.Series([output] * len(data))  # Broadcast scalar to match data length
#         except Exception as e:
#             results_dict[name] = pd.Series([f"Error: {str(e)}"] * len(data))  # Store errors in the DataFrame

#     return pd.DataFrame(results_dict, index=data.index)

# # Load functions from the script directory
# script_folder = "generated_strategies"  # Update with your actual folder path
# functions = load_functions_from_directory(script_folder)

# # Load dataset
# file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
# data = pd.read_csv(file_path, index_col=0)

# # Ensure index is in datetime format
# data.index = pd.to_datetime(data.index)

# # Call functions with the dataset and store in DataFrame
# results_df = run_all_functions(functions, data)

# # Print first few rows for verification
# print(results_df.head())

# # %%

# %%
