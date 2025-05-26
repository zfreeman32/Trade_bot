import pandas as pd
import numpy as np
import time
import cupy as cp  # GPU acceleration library
from numba import jit, cuda, prange  # For CPU and GPU acceleration
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Performance timing decorator
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

# ---------------------------
# 1. Load and prepare data
# ---------------------------
@timer
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

# ---------------------------
# 2. Compute ATR and Volume Moving Average
# ---------------------------
@jit(nopython=True, parallel=True)  # JIT compilation for CPU acceleration
def compute_tr_fast(high, low, close):
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]  # First row
    
    for i in prange(1, n):  # Use parallel range for multi-core
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)
    
    return tr

@timer
def compute_indicators(df, atr_period=14, vol_ma_period=20):
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    # Calculate True Range using JIT-accelerated function
    tr = compute_tr_fast(high, low, close)
    
    # Calculate ATR as rolling mean of TR
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate volume moving average
    volumes = df['Volume'].values
    vol_ma = np.zeros_like(volumes)
    for i in range(len(volumes)):
        if i < vol_ma_period:
            vol_ma[i] = np.mean(volumes[:i+1])
        else:
            vol_ma[i] = np.mean(volumes[i-vol_ma_period+1:i+1])
    
    df['ATR'] = atr
    df['vol_ma'] = vol_ma
    
    return df, close, atr, vol_ma, volumes

# ---------------------------
# 3. Dynamic Programming for Trade Signals - GPU Version
# ---------------------------
# CUDA kernel for finding trades
@cuda.jit
def find_trades_gpu(prices, atr, volumes, vol_ma, min_profit, atr_mult, vol_mult, is_long, results, trade_indices):
    i = cuda.grid(1)
    if i >= len(prices):
        return
    
    # Only consider if volume is above average
    if volumes[i] < vol_mult * vol_ma[i]:
        results[i] = 0
        trade_indices[i, 0] = -1
        trade_indices[i, 1] = -1
        return
    
    best_profit = 0
    best_exit = -1
    
    for j in range(i + 1, len(prices)):
        # For long trades, we want rising prices; for short trades, we want falling prices
        profit = (prices[j] - prices[i]) if is_long else (prices[i] - prices[j])
        
        if profit >= min_profit and profit >= atr_mult * atr[i]:
            if profit > best_profit:
                best_profit = profit
                best_exit = j
    
    results[i] = best_profit
    trade_indices[i, 0] = i if best_exit != -1 else -1
    trade_indices[i, 1] = best_exit

@timer
def process_trades_gpu(prices, atr, volumes, vol_ma, min_profit_threshold=0.0015, 
                        atr_multiplier=1.0, volume_multiplier=1.0, is_long=True):
    # Transfer data to GPU
    d_prices = cp.asarray(prices)
    d_atr = cp.asarray(atr)
    d_volumes = cp.asarray(volumes)
    d_vol_ma = cp.asarray(vol_ma)
    
    # Prepare output arrays
    d_results = cp.zeros_like(d_prices)
    d_trade_indices = cp.full((len(prices), 2), -1, dtype=cp.int32)
    
    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = (len(prices) + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    find_trades_gpu[blocks_per_grid, threads_per_block](
        d_prices, d_atr, d_volumes, d_vol_ma, 
        min_profit_threshold, atr_multiplier, volume_multiplier,
        is_long, d_results, d_trade_indices
    )
    
    # Transfer results back to CPU
    trade_indices = cp.asnumpy(d_trade_indices)
    
    # Filter valid trades
    valid_trades = []
    for i in range(len(trade_indices)):
        if trade_indices[i, 0] != -1 and trade_indices[i, 1] != -1:
            valid_trades.append((int(trade_indices[i, 0]), int(trade_indices[i, 1])))
    
    # Sort and remove overlapping trades
    valid_trades.sort(key=lambda x: x[0])
    non_overlapping_trades = []
    
    if valid_trades:
        current_trade = valid_trades[0]
        non_overlapping_trades.append(current_trade)
        
        for trade in valid_trades[1:]:
            if trade[0] > current_trade[1]:
                non_overlapping_trades.append(trade)
                current_trade = trade
    
    return non_overlapping_trades

# CPU Fallback version (using Numba for acceleration)
@jit(nopython=True)
def find_trades_cpu(prices, atr, volumes, vol_ma, min_profit_threshold, 
                    atr_multiplier, volume_multiplier, is_long):
    n = len(prices)
    dp = np.zeros(n + 1)
    trade_decision = np.zeros((n, 2), dtype=np.int32) - 1
    
    # Iterate backwards
    for i in range(n - 1, -1, -1):
        best_profit = dp[i + 1]
        best_entry = -1
        best_exit = -1
        
        # Only consider if volume is above average
        if volumes[i] < volume_multiplier * vol_ma[i]:
            dp[i] = best_profit
            continue
        
        # Test potential exit points j > i
        for j in range(i + 1, n):
            profit_trade = (prices[j] - prices[i]) if is_long else (prices[i] - prices[j])
            
            # Must exceed thresholds
            if profit_trade < min_profit_threshold:
                continue
            if profit_trade < atr_multiplier * atr[i]:
                continue
            
            total_profit = profit_trade + (dp[j + 1] if (j + 1) < len(dp) else 0)
            if total_profit > best_profit:
                best_profit = total_profit
                best_entry = i
                best_exit = j
                
        dp[i] = best_profit
        if best_entry != -1:
            trade_decision[i, 0] = best_entry
            trade_decision[i, 1] = best_exit
    
    # Extract non-overlapping trades
    trades = []
    i = 0
    while i < n:
        if trade_decision[i, 0] != -1:
            trades.append((int(trade_decision[i, 0]), int(trade_decision[i, 1])))
            i = trade_decision[i, 1] + 1
        else:
            i += 1
            
    return trades

@timer
def process_trades_cpu(prices, atr, volumes, vol_ma, min_profit_threshold=0.0015, 
                       atr_multiplier=1.0, volume_multiplier=1.0, is_long=True):
    return find_trades_cpu(prices, atr, volumes, vol_ma, min_profit_threshold, 
                          atr_multiplier, volume_multiplier, is_long)

# ---------------------------
# 4. Merge and Assign Signals - Vectorized approach
# ---------------------------
@timer
def merge_and_assign_signals(df, long_trades, short_trades):
    # Create signal columns
    df['long_signal'] = 0
    df['short_signal'] = 0
    df['close_position'] = 0
    
    # Combine and sort all trades
    all_trades = []
    for trade in long_trades:
        all_trades.append((trade[0], trade[1], 'long'))
    for trade in short_trades:
        all_trades.append((trade[0], trade[1], 'short'))
    
    all_trades.sort(key=lambda x: x[0])
    
    # Use vectorized operations where possible
    for i, trade in enumerate(all_trades):
        entry_idx, exit_idx, ttype = trade
        
        if ttype == 'long':
            df.loc[entry_idx, 'long_signal'] = 1
        else:
            df.loc[entry_idx, 'short_signal'] = 1
        
        # Close the previous trade
        if i < len(all_trades) - 1:
            next_entry = all_trades[i + 1][0]
            close_idx = next_entry - 1 if next_entry - 1 > entry_idx else exit_idx
            df.loc[close_idx, 'close_position'] = 1
        else:
            df.loc[exit_idx, 'close_position'] = 1
    
    return df

# ---------------------------
# Main execution function
# ---------------------------
@timer
def main():
    # RunPod specific - detect GPU and set optimal thread count
    device_count = cp.cuda.runtime.getDeviceCount()
    cpu_count = mp.cpu_count()
    print(f"Detected {device_count} CUDA device(s) and {cpu_count} CPU cores")
    
    # Allow specifying input files as arguments or use defaults
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'EURUSD_1min_sampled.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else "EURUSD_1min_sampled_signals.csv"
    
    print(f"Processing {input_file} -> {output_file}")
    
    print("Loading and preparing data...")
    df = load_data(input_file)
    
    print("Computing indicators...")
    df, prices, atr, vol_ma, volumes = compute_indicators(df)
    
    print("Finding trades...")
    try:
        # Try GPU first
        print("Using GPU acceleration...")
        long_trades = process_trades_gpu(prices, atr, volumes, vol_ma, 
                                         min_profit_threshold=0.0015, is_long=True)
        short_trades = process_trades_gpu(prices, atr, volumes, vol_ma, 
                                          min_profit_threshold=0.0015, is_long=False)
    except Exception as e:
        print(f"GPU processing failed with error: {e}")
        print("Falling back to CPU processing...")
        long_trades = process_trades_cpu(prices, atr, volumes, vol_ma, 
                                         min_profit_threshold=0.0015, is_long=True)
        short_trades = process_trades_cpu(prices, atr, volumes, vol_ma, 
                                          min_profit_threshold=0.0015, is_long=False)
    
    print(f"Found {len(long_trades)} long trades and {len(short_trades)} short trades")
    
    print("Merging and assigning signals...")
    df = merge_and_assign_signals(df, long_trades, short_trades)
    
    print("Saving results...")
    df = df.drop(columns=['ATR', 'vol_ma'])
    df.to_csv(output_file, index=False)
    
    return df

if __name__ == "__main__":
    try:
        # Print system info
        import os
        import psutil
        
        # Get GPU info
        try:
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            print(f"GPU: {gpu_info['name'].decode()}")
            print(f"GPU Memory: {gpu_info['totalGlobalMem'] / (1024**3):.2f} GB")
        except:
            print("Unable to get detailed GPU info")
            
        # Get CPU info
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count()} Logical")
        print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
        
        # Show current directory and files
        print(f"Working Directory: {os.getcwd()}")
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        if csv_files:
            print(f"Available CSV files: {', '.join(csv_files)}")
        else:
            print("Warning: No CSV files found in current directory")
            
        # Run main function
        main()
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()