import os
import re
import sys
import pandas as pd
import numpy as np
import importlib.util
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# Performance timer decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

@timer
def load_strategies(module_path):
    """Dynamically loads all functions from the given module path."""
    print(f"Loading strategies from {module_path}")
    spec = importlib.util.spec_from_file_location("all_strategies", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["all_strategies"] = module
    spec.loader.exec_module(module)
    return {name: func for name, func in vars(module).items() 
            if callable(func) and not name.startswith('_')}

@timer
def load_data(file_path):
    """Load and prepare data efficiently."""
    print(f"Loading data from {file_path}")
    
    # Use low_memory=False to prevent mixed type inference
    stock_df = pd.read_csv(file_path, dtype={'Date': str, 'Time': str}, low_memory=False)
    
    # Convert Date & Time to Datetime - use format for faster conversion
    stock_df['Datetime'] = pd.to_datetime(stock_df['Date'] + ' ' + stock_df['Time'], 
                                         format='%Y%m%d %H:%M:%S', errors='coerce')
    
    # Check for parsing errors
    if stock_df['Datetime'].isna().any():
        print("Warning: Some datetime values couldn't be parsed!")
    
    # Set Datetime as index
    stock_df.set_index('Datetime', inplace=True)
    
    # Optimize memory usage
    for col in stock_df.select_dtypes(include=['float']):
        stock_df[col] = pd.to_numeric(stock_df[col], downcast='float')
    
    for col in stock_df.select_dtypes(include=['int']):
        stock_df[col] = pd.to_numeric(stock_df[col], downcast='integer')
    
    return stock_df

def apply_strategy_safe(strategy_name, strategy_func, stock_df, chunk_size=None):
    """Safely apply a strategy function with error handling."""
    try:
        # Use a smaller chunk if specified (for parallel processing)
        df_to_use = stock_df.iloc[:chunk_size] if chunk_size else stock_df
        
        # Apply the strategy function
        start_time = time.time()
        result = strategy_func(df_to_use)
        end_time = time.time()
        
        # Check the result
        if isinstance(result, pd.DataFrame) and not result.empty:
            print(f"Strategy {strategy_name} completed in {end_time - start_time:.4f} seconds")
            # Return only new columns to save memory
            return result
        else:
            print(f"Strategy {strategy_name} returned empty or invalid result")
            return None
    except Exception as e:
        print(f"Error applying {strategy_name}: {e}")
        return None

@timer
def apply_strategies_parallel(strategies, stock_df, max_workers=None):
    """Apply strategies in parallel using ProcessPoolExecutor."""
    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 1)  # Leave one core free
    
    print(f"Applying {len(strategies)} strategies using {max_workers} workers")
    results = {}
    
    # Define a worker function that takes a strategy name and function
    def worker(name_func_pair):
        name, func = name_func_pair
        return name, apply_strategy_safe(name, func, stock_df)
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, (name, func)) 
                  for name, func in strategies.items()]
        
        for future in futures:
            try:
                name, result = future.result()
                if result is not None:
                    results[name] = result
            except Exception as e:
                print(f"Worker failed: {e}")
    
    return results

@timer
def merge_results(results_dict, base_df):
    """Efficiently merge results from multiple strategies."""
    print("Merging results from all strategies")
    signals_df = base_df.copy()
    
    # Get a list of all unique columns across all results
    all_new_cols = set()
    for result_df in results_dict.values():
        if result_df is not None:
            all_new_cols.update([col for col in result_df.columns 
                                if col not in signals_df.columns])
    
    # Initialize all new columns with NaN to avoid reindexing issues
    for col in all_new_cols:
        signals_df[col] = np.nan
    
    # Fill in values from each result
    for strategy_name, result_df in results_dict.items():
        if result_df is None:
            continue
        
        print(f"Adding columns from {strategy_name}")
        new_cols = [col for col in result_df.columns if col not in base_df.columns]
        
        for col in new_cols:
            try:
                # Align by index to handle length mismatches
                signals_df.loc[result_df.index, col] = result_df[col]
            except Exception as e:
                print(f"Error adding column {col} from {strategy_name}: {e}")
    
    return signals_df

@timer
def clean_dataframe(df):
    """Clean up the dataframe by removing duplicates, constants, etc."""
    original_shape = df.shape
    print(f"Original DataFrame shape: {original_shape}")
    
    # Remove duplicate columns (more efficient than checking pairs)
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"After removing duplicate columns: {df.shape}")
    
    # Remove constant columns (optimized approach)
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)
    print(f"After removing {len(constant_cols)} constant columns: {df.shape}")
    
    # Remove columns with too many missing values
    threshold = 0.5 * len(df)
    missing_values = df.isnull().sum()
    cols_to_drop = missing_values[missing_values > threshold].index
    df = df.drop(columns=cols_to_drop)
    print(f"After removing {len(cols_to_drop)} columns with >50% missing values: {df.shape}")
    
    return df

@timer
def analyze_dataframe(df):
    """Perform basic analysis on the dataframe."""
    # Print shape
    print(f"Shape of DataFrame: {df.shape}")
    
    # Check for duplicate columns (optimized approach)
    duplicate_count = df.shape[1] - df.loc[:, ~df.columns.duplicated()].shape[1]
    print(f"Duplicate columns: {duplicate_count}")
    
    # Identify constant columns
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    print(f"Constant columns: {len(constant_cols)}")
    
    # Check for NaN values
    missing_values = df.isnull().sum()
    missing_count = (missing_values > 0).sum()
    print(f"Columns with missing values: {missing_count}")
    
    # Check for duplicate rows (using a sample for large datasets)
    if len(df) > 100000:
        sample_size = min(100000, int(len(df) * 0.1))
        sample_df = df.sample(sample_size)
        duplicate_rows = sample_df.duplicated().sum()
        print(f"Duplicate rows in sample: {duplicate_rows} (sampled {sample_size} rows)")
    else:
        duplicate_rows = df.duplicated().sum()
        print(f"Duplicate rows: {duplicate_rows}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Memory usage: {memory_usage:.2f} MB")

@timer
def main():
    # Configuration
    strategies_file = "all_strategies.py"
    file_path = 'EURUSD_1min_sampled_signals.csv'
    output_file = 'EURUSD_1min_sampled_features.csv'
    
    # Use as many cores as available, minus one for system
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Load data
    stock_df = load_data(file_path)
    
    # Load strategies
    strategies = load_strategies(strategies_file)
    print(f"Loaded {len(strategies)} strategies")
    
    # Import indicators module and generate indicators
    print("Generating indicators...")
    try:
        from all_indicators import generate_all_indicators
        start_time = time.time()
        indicators_df = generate_all_indicators(stock_df)
        end_time = time.time()
        print(f"Generated indicators in {end_time - start_time:.4f} seconds")
    except Exception as e:
        print(f"Error generating indicators: {e}")
        indicators_df = stock_df.copy()
    
    # Apply strategies in parallel
    strategy_results = apply_strategies_parallel(strategies, stock_df, max_workers=num_cores)
    
    # Merge results
    signals_df = merge_results(strategy_results, indicators_df)
    
    # Analyze the dataframe
    analyze_dataframe(signals_df)
    
    # Clean the dataframe
    signals_df = clean_dataframe(signals_df)
    
    # Save as CSV
    print(f"Saving to {output_file}...")
    signals_df.to_csv(output_file, index=False)
    print(f"CSV file saved as {output_file}")
    
    return signals_df

if __name__ == "__main__":
    try:
        # Print system info
        print("=" * 50)
        print("SYSTEM INFORMATION")
        print("=" * 50)
        
        # CPU information
        cpu_count = mp.cpu_count()
        print(f"CPU Cores: {cpu_count}")
        
        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            print(f"Memory: Total={memory.total/(1024**3):.1f}GB, Available={memory.available/(1024**3):.1f}GB")
        except ImportError:
            print("psutil not available for memory information")
        
        # Python version
        print(f"Python Version: {sys.version}")
        
        # pandas version
        print(f"Pandas Version: {pd.__version__}")
        
        # NumPy version
        print(f"NumPy Version: {np.__version__}")
        
        print("=" * 50)
        
        # Set pandas options for better performance
        pd.set_option('mode.chained_assignment', None)
        pd.options.mode.use_inf_as_na = True
        
        # Run main function
        main()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()