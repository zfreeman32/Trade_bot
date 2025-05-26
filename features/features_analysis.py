#%%
import sys
import traceback
import pandas as pd
import numpy as np
from scipy import stats
import shap
import tqdm
import time
import psutil
import os
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Add GPU support with RAPIDS libraries
try:
    import cudf
    print("Successfully imported cudf")
    import cupy as cp
    print("Successfully imported cupy")
    from cuml.ensemble import RandomForestClassifier as cuRF_Classifier
    from cuml.ensemble import RandomForestRegressor as cuRF_Regressor
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from cuml.model_selection import train_test_split as cu_train_test_split
    print("Successfully imported cuml")
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with RAPIDS")
except ImportError:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.utils.multiclass import type_of_target
    from sklearn.metrics import make_scorer
    GPU_AVAILABLE = False
    print("RAPIDS not available, using CPU implementation")

# Import sklearn libraries anyway for fallback and operations not supported in cuML
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

# Initialize global variables for timing
start_time = time.time()
section_times = {}
current_section = None
section_start_time = None

# Override print function to include timestamps
original_print = print
def print(*args, **kwargs):
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elapsed = time.time() - start_time
    elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    original_print(f"[{current_time}] (+{elapsed_formatted})", *args, **kwargs)

# Define log levels
LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3
CURRENT_LOG_LEVEL = LOG_LEVEL_INFO  # Set default log level

# Default target settings - will be overridden in configurations
target_col = 'long_signal'
regression_mode = False

def write_partial_results(results, section_name, file_path=None, mode='a'):
    """
    Write partial results to file as they become available
    
    Args:
        results: Dictionary of results to write
        section_name: Name of the current section
        file_path: Path to output file
        mode: File open mode ('w' for new file, 'a' for append)
    """
    if file_path is None:
        file_path = f"{target_col}_analysis_results.txt"
    
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(file_path, mode) as f:
        # If this is the first write, add a header
        if mode == 'w':
            f.write(f"=== ENHANCED TRADING DATASET ANALYSIS ===\n")
            f.write(f"Started: {current_time}\n\n")
        
        # Write section header with timestamp
        f.write(f"\n{'='*20} {section_name} - {current_time} {'='*20}\n\n")
        
        if section_name == "Dataset Structure":
            for key, value in results['dataset_structure'].items():
                f.write(f"{key}: {value}\n")
        
        elif section_name == "Feature Statistics":
            # Sort features by correlation with target
            sorted_features = sorted(
                results['feature_statistics'].items(),
                key=lambda x: abs(x[1].get('corr_with_target', 0)), 
                reverse=True
            )
            f.write("Top 20 Features by Target Correlation:\n" + "-" * 50 + "\n")
            for feature, stats in sorted_features[:20]:
                f.write(f"{feature}:\n")
                for stat, value in stats.items():
                    f.write(f"  {stat}: {value:.4f}\n")
        
        elif section_name == "Periodic Patterns":
            for lag, correlation in results['periodic_patterns'].items():
                f.write(f"{lag}: {correlation:.4f}\n")
        
        elif section_name == "Feature Importance":
            if 'mutual_information' in results['feature_importance']:
                f.write("\nMutual Information Scores (Top 20):\n")
                for i, (feature, score) in enumerate(list(results['feature_importance']['mutual_information'].items())[:20]):
                    f.write(f"{i+1}. {feature}: {score:.6f}\n")
            
            if 'random_forest_importance' in results['feature_importance']:
                f.write("\nRandom Forest Feature Importance (Top 20):\n")
                for i, (feature, score) in enumerate(list(results['feature_importance']['random_forest_importance'].items())[:20]):
                    f.write(f"{i+1}. {feature}: {score:.6f}\n")
                
            if 'direction_importance' in results['feature_importance']:
                f.write("\nDirectional Accuracy Importance (Top 20):\n")
                for i, (feature, score) in enumerate(list(results['feature_importance']['direction_importance'].items())[:20]):
                    f.write(f"{i+1}. {feature}: {score:.6f}\n")
        
        elif section_name == "SHAP Analysis":
            if 'shap_importance' in results['shap_analysis'] and results['shap_analysis']['shap_importance']:
                f.write("\nSHAP Feature Importance (Top 20):\n")
                for i, (feature, score) in enumerate(list(results['shap_analysis']['shap_importance'].items())[:20]):
                    f.write(f"{i+1}. {feature}: {score:.6f}\n")
            else:
                f.write("\nSHAP analysis not completed or failed.\n")
        
        elif section_name == "Time Series Stability":
            f.write(f"{'Feature':<30}{'Mean Importance':<15}{'Stability (CV)':<15}{'Trend':<15}\n")
            f.write("-" * 75 + "\n")
            
            sorted_stability = sorted(
                results['time_series_stability'].items(),
                key=lambda x: x[1]['mean_importance'],
                reverse=True
            )
            
            for feature, metrics in sorted_stability[:20]:
                f.write(f"{feature:<30}{metrics['mean_importance']:<15.4f}{metrics['coefficient_of_variation']:<15.4f}{metrics['trend']:<15.4e}\n")
                
        elif section_name == "Important Features":
            important_features = set()
            
            # Consider features important based on threshold method - similar to write_results_to_file
            importance_methods = [
                ('mutual_information', results.get('feature_importance', {})), 
                ('random_forest_importance', results.get('feature_importance', {})),
                ('shap_importance', results.get('shap_analysis', {}))
            ]
            
            for method_name, method_dict in importance_methods:
                if method_name in method_dict:
                    values = list(method_dict[method_name].values())
                    # Use mean + 1 std as threshold
                    if values:
                        threshold = np.mean(values) + np.std(values)
                        for feature, score in method_dict[method_name].items():
                            if score > threshold:
                                important_features.add(feature)
                                
            # Include directional importance if available
            if 'direction_importance' in results.get('feature_importance', {}):
                values = list(results['feature_importance']['direction_importance'].values())
                threshold = np.mean(values) + np.std(values)
                for feature, score in results['feature_importance']['direction_importance'].items():
                    if score > threshold:
                        important_features.add(feature)
            
            # Also consider time-series stability
            stable_features = []
            if 'time_series_stability' in results:
                for feature, metrics in results['time_series_stability'].items():
                    # Features with low coefficient of variation are more stable
                    if metrics['coefficient_of_variation'] < 0.5:  # Threshold for stability
                        stable_features.append((feature, metrics['mean_importance']))
            
            # Sort stable features by importance
            stable_features.sort(key=lambda x: x[1], reverse=True)
            
            # Add top stable features to important_features
            for feature, _ in stable_features[:min(20, len(stable_features))]:
                important_features.add(feature)

            f.write("=== IMPORTANT FEATURES ===\n\n")
            f.write("# Features with high importance and stability:\n")
            for feature in sorted(important_features):
                f.write(feature + '\n')
            
            f.write("\n# Top stable features across time periods:\n")
            for feature, importance in stable_features[:20]:
                f.write(f"{feature}: {importance:.4f}\n")
        
        f.write("\n")  # Add spacing at end of section

def log(message, level=LOG_LEVEL_INFO, *args, **kwargs):
    """Enhanced logging function with timing and memory info"""
    if level < CURRENT_LOG_LEVEL:
        return
        
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    elapsed = time.time() - start_time
    elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed))
    
    # Get memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024
    
    # Format the prefix based on log level
    level_prefix = ""
    if level == LOG_LEVEL_DEBUG:
        level_prefix = "[DEBUG] "
    elif level == LOG_LEVEL_WARNING:
        level_prefix = "[WARNING] "
    elif level == LOG_LEVEL_ERROR:
        level_prefix = "[ERROR] "
    
    # Include section information if available
    section_info = f"[{current_section}] " if current_section else ""
    
    # Print with timing and memory info
    original_print(f"[{current_time}] (+{elapsed_formatted}) {level_prefix}{section_info}MEM:{memory_usage_mb:.1f}MB - {message}", *args, **kwargs)
    
    # Add GPU memory tracking if available
    if GPU_AVAILABLE:
        try:
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            gpu_memory_used = (total_memory - free_memory) / 1024 / 1024
            gpu_utilization = 100 * (1 - free_memory / total_memory)
            original_print(f"    GPU Memory: {gpu_memory_used:.1f}MB ({gpu_utilization:.1f}% used)")
        except:
            pass

def start_section(section_name):
    """Mark the beginning of a new code section for timing"""
    global current_section, section_start_time
    
    # If we were in another section, log its completion
    if current_section:
        end_section()
    
    current_section = section_name
    section_start_time = time.time()
    log(f"Starting section: {section_name}")

def end_section():
    """Mark the end of the current code section"""
    global current_section, section_start_time, section_times
    
    if current_section:
        elapsed = time.time() - section_start_time
        section_times[current_section] = elapsed
        log(f"Completed section: {current_section} in {elapsed:.2f} seconds")
        current_section = None
        section_start_time = None

def log_progress(current, total, message="Progress", frequency=5):
    """Log progress for long-running operations"""
    if total == 0:
        percentage = 100
    else:
        percentage = (current / total) * 100
        
    # Only log at certain percentage intervals to avoid log spam
    if current == 0 or current == total-1 or current % max(1, total // frequency) == 0:
        log(f"{message}: {current+1}/{total} ({percentage:.1f}%)", LOG_LEVEL_DEBUG)

def print_timing_summary():
    """Print a summary of time spent in each section"""
    log("=== TIMING SUMMARY ===")
    sorted_sections = sorted(section_times.items(), key=lambda x: x[1], reverse=True)
    total_time = sum(section_times.values())
    
    for section, elapsed in sorted_sections:
        percentage = (elapsed / total_time) * 100
        log(f"{section}: {elapsed:.2f}s ({percentage:.1f}%)")
    
    log(f"Total execution time: {total_time:.2f}s")

def timing_decorator(func):
    """Decorator to time function calls"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_section(func_name)
        result = func(*args, **kwargs)
        end_section()
        return result
    return wrapper

# Custom direction-aware loss function for regression
def direction_aware_loss(y_true, y_pred, direction_weight=0.3, magnitude_weight=0.7):
    """
    Custom loss that balances absolute prediction error with directional accuracy
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        direction_weight: Weight for directional accuracy (0-1)
        magnitude_weight: Weight for magnitude accuracy (0-1)
    
    Returns:
        Loss value combining directional and magnitude components
    """
    # Calculate if direction was predicted correctly (1 if correct, 0 if wrong)
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    direction_correct = (true_direction == pred_direction).astype(float)
    
    # Directional accuracy component (higher is better)
    direction_score = np.mean(direction_correct)
    
    # Magnitude accuracy component (lower is better)
    magnitude_error = np.mean(np.abs(y_true - y_pred))
    
    # Normalize magnitude error to 0-1 range approximately
    normalized_magnitude_error = np.tanh(magnitude_error / np.std(y_true))
    
    # Combined loss (lower is better)
    combined_loss = (direction_weight * (1 - direction_score)) + \
                   (magnitude_weight * normalized_magnitude_error)
    
    return combined_loss

# Function to evaluate directional accuracy
def direction_accuracy(y_true, y_pred):
    """Calculate percentage of correct directional predictions"""
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    return np.mean(true_direction == pred_direction)

# Create scorer for sklearn models
direction_scorer = make_scorer(direction_accuracy, greater_is_better=True)

# Function to create profit-aware targets for classification
def create_profit_aware_targets(df, signal_col, profit_window=14, stop_loss_pct=0.01):
    """
    Creates profit-aware target by checking if signal would have led to profit
    
    Args:
        df: DataFrame with price data
        signal_col: Column with original signals (0/1)
        profit_window: Number of periods to check for profit
        stop_loss_pct: Stop loss percentage (0.01 = 1%)
    
    Returns:
        Series with profit-aware target (1 if signal was profitable, 0 otherwise)
    """
    if signal_col == 'long_signal':
        # For long signals, check if price increased
        df['entry_price'] = df['Close'].shift(-1)  # Entry at next bar's close
        df['exit_price'] = df['Close'].shift(-profit_window)  # Exit after profit_window
        
        # Calculate stop loss price
        df['stop_price'] = df['entry_price'] * (1 - stop_loss_pct)
        
        # Check if stop was hit before profit target
        hit_stop = False
        for i in range(1, profit_window):
            # If any Low price between entry and exit is below stop price
            df[f'hit_stop_{i}'] = df['Low'].shift(-i) < df['stop_price']
            hit_stop = hit_stop | df[f'hit_stop_{i}']
            
        # Calculate PnL (exit_price - entry_price)
        df['pnl'] = df['exit_price'] - df['entry_price']
        
        # Signal is profitable if PnL > 0 and stop loss wasn't hit
        df['profitable_signal'] = ((df['pnl'] > 0) & ~hit_stop & (df[signal_col] == 1)).astype(int)
        
    elif signal_col == 'short_signal':
        # For short signals, check if price decreased
        df['entry_price'] = df['Close'].shift(-1)  # Entry at next bar's close
        df['exit_price'] = df['Close'].shift(-profit_window)  # Exit after profit_window
        
        # Calculate stop loss price for shorts
        df['stop_price'] = df['entry_price'] * (1 + stop_loss_pct)
        
        # Check if stop was hit before profit target
        hit_stop = False
        for i in range(1, profit_window):
            # If any High price between entry and exit is above stop price
            df[f'hit_stop_{i}'] = df['High'].shift(-i) > df['stop_price']
            hit_stop = hit_stop | df[f'hit_stop_{i}']
            
        # Calculate PnL (entry_price - exit_price) for shorts
        df['pnl'] = df['entry_price'] - df['exit_price']
        
        # Signal is profitable if PnL > 0 and stop loss wasn't hit
        df['profitable_signal'] = ((df['pnl'] > 0) & ~hit_stop & (df[signal_col] == 1)).astype(int)
    
    # Clean up temporary columns
    temp_cols = ['entry_price', 'exit_price', 'stop_price', 'pnl'] + [f'hit_stop_{i}' for i in range(1, profit_window)]
    df.drop(columns=temp_cols, inplace=True)
    
    return df['profitable_signal']

class EnhancedTradingDataAnalyzer:
    def __init__(self, df, target_col=target_col, regression_mode=regression_mode, forecast_periods=14):
        self.df = df.copy()
        self.original_target_col = target_col
        self.regression_mode = regression_mode
        self.forecast_periods = forecast_periods
        self.feature_stats = {}
        
        # Use GPU for processing if available
        self.use_gpu = GPU_AVAILABLE
        
        # Prepare targets based on mode
        if regression_mode:
            # For regression, create both absolute Close and directional targets
            self.df['future_close'] = self.df['Close'].shift(-forecast_periods)
            self.df['pct_change'] = self.df['future_close'].pct_change(periods=forecast_periods)
            self.df['direction'] = np.sign(self.df['pct_change'])
            self.target_col = 'future_close'
            
            # Remove NaN rows from the derived targets
            self.df = self.df.dropna(subset=[self.target_col, 'pct_change', 'direction'])
        else:
            # For classification, create profit-aware targets
            self.df['profitable_signal'] = create_profit_aware_targets(
                self.df, target_col, profit_window=forecast_periods)
            self.target_col = 'profitable_signal'
            
            # Remove NaN rows from the derived targets
            self.df = self.df.dropna(subset=[self.target_col])
        
        # Remove constant features that cause correlation issues
        self._remove_constant_features()
        
        # Convert to GPU dataframe if GPU available
        if self.use_gpu:
            try:
                log("Converting dataframe to GPU memory")
                self.gpu_df = cudf.DataFrame.from_pandas(self.df)
                log(f"DataFrame successfully moved to GPU memory")
            except Exception as e:
                log(f"Failed to move dataframe to GPU: {e}", LOG_LEVEL_WARNING)
                self.use_gpu = False
                self.gpu_df = None

    def _remove_constant_features(self, threshold=1e-10):
        """Remove features with near-zero standard deviation"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        constant_features = []
        for col in numeric_cols:
            if self.df[col].std() <= threshold:
                constant_features.append(col)
        
        if constant_features:
            log(f"Removed {len(constant_features)} constant features that would cause correlation issues")
            log(", ".join(constant_features[:10]) + 
                (f" and {len(constant_features)-10} more..." if len(constant_features) > 10 else ""))
            self.df = self.df.drop(columns=constant_features)

    def analyze_dataset_structure(self):
        start_section("analyze_dataset_structure")
        log("Analyzing dataset structure")
        
        analysis = {
            'total_samples': len(self.df),
            'feature_count': len(self.df.columns),
            'memory_usage_MB': self.df.memory_usage().sum() / 1024**2,
            'missing_values': self.df.isnull().sum().sum(),
            'datatypes': self.df.dtypes.value_counts().to_dict(),
            'gpu_acceleration': "Enabled" if self.use_gpu else "Disabled"
        }

        if self.regression_mode:
            # Regression stats
            analysis['target_stats'] = {
                'mean': float(self.df[self.target_col].mean()),
                'std': float(self.df[self.target_col].std()),
                'min': float(self.df[self.target_col].min()),
                'max': float(self.df[self.target_col].max()),
                'median': float(self.df[self.target_col].median()),
                'direction_distribution': {
                    'up': int((self.df['direction'] > 0).sum()),
                    'down': int((self.df['direction'] < 0).sum()),
                    'unchanged': int((self.df['direction'] == 0).sum()),
                }
            }
        else:
            # Classification stats
            original_signal_counts = self.df[self.original_target_col].value_counts()
            profitable_signal_counts = self.df[self.target_col].value_counts()
            
            total = len(self.df)
            original_signal_ratio = original_signal_counts.get(1, 0) / total
            profitable_signal_ratio = profitable_signal_counts.get(1, 0) / total
            
            # Calculate what % of original signals were profitable
            signals_count = original_signal_counts.get(1, 0)
            if signals_count > 0:
                profit_rate = self.df[self.df[self.original_target_col] == 1][self.target_col].mean()
            else:
                profit_rate = 0

            analysis['signal_distribution'] = {
                'original_signals': int(original_signal_counts.get(1, 0)),
                'profitable_signals': int(profitable_signal_counts.get(1, 0)),
                'no_original_signals': int(original_signal_counts.get(0, 0)),
                'no_profitable_signals': int(profitable_signal_counts.get(0, 0)),
                'original_signal_ratio': original_signal_ratio,
                'profitable_signal_ratio': profitable_signal_ratio,
                'profit_rate': profit_rate,  # % of signals that were profitable
                'imbalance_warning': profitable_signal_ratio < 0.1 or profitable_signal_ratio > 0.9,
            }
            
        end_section()
        return analysis

    def analyze_feature_statistics(self):
        start_section("analyze_feature_statistics")
        log("Computing feature statistics")
        
        # We'll use CPU for this as it's not computationally intensive
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        
        # Process features in parallel using ThreadPoolExecutor
        def process_feature(feature):
            if feature in [self.target_col, self.original_target_col, 'future_close', 'pct_change', 'direction', 'profitable_signal']:
                return None
                
            feature_data = self.df[feature].dropna()
            
            # Skip features with not enough data
            if len(feature_data) < 2:
                return None
                
            stats_data = {
                'mean': self.df[feature].mean(),
                'median': self.df[feature].median(),
                'std': self.df[feature].std(),
                'skew': stats.skew(feature_data),
                'kurtosis': stats.kurtosis(feature_data),
                'unique_values': self.df[feature].nunique(),
                'iqr': self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25),
                'outlier_percentage': np.mean((self.df[feature] < self.df[feature].quantile(0.25) - 1.5 * (self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25))) |
                                           (self.df[feature] > self.df[feature].quantile(0.75) + 1.5 * (self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25)))) * 100
            }
            
            # Add correlation with target
            if self.regression_mode:
                stats_data['corr_with_target'] = self.df[feature].corr(self.df[self.target_col])
                stats_data['corr_with_direction'] = self.df[feature].corr(self.df['direction'])
            else:
                stats_data['corr_with_target'] = self.df[feature].corr(self.df[self.target_col])
                stats_data['corr_with_original_signal'] = self.df[feature].corr(self.df[self.original_target_col])
                
            return (feature, stats_data)
        
        # Use parallel processing for feature statistics (CPU-bound)
        num_workers = min(os.cpu_count(), 12)  # Use up to 12 cores
        log(f"Processing feature statistics with {num_workers} workers")
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_feature, numeric_features))
        
        # Process results
        for result in results:
            if result is not None:
                feature, stats_data = result
                self.feature_stats[feature] = stats_data
                
        log(f"Completed statistics for {len(self.feature_stats)} features")
        end_section()
        return self.feature_stats

    def analyze_periodic_patterns(self):
        start_section("analyze_periodic_patterns")
        log("Analyzing periodic patterns")
        
        # This is a simple operation, no need for GPU
        autocorr_values = {f'lag_{lag}': self.df[self.target_col].autocorr(lag) 
                           for lag in range(1, min(100, len(self.df) // 2))}
        top_correlations = sorted(autocorr_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        end_section()
        return {key: val for key, val in top_correlations}

    def _calculate_mutual_information(self, X, y, numeric_features, use_gpu=False):
        """Calculate mutual information scores with GPU if available"""
        log("Calculating mutual information scores")
        
        if use_gpu and self.use_gpu:
            try:
                if self.regression_mode:
                    mi_scores = mutual_info_regression(X, y)
                else:
                    mi_scores = mutual_info_classif(X, y)
                    
                return {feature: float(score) for feature, score in zip(numeric_features, mi_scores)}
            except Exception as e:
                log(f"GPU mutual info calculation failed, falling back to CPU: {e}", LOG_LEVEL_WARNING)
        
        # CPU fallback
        if self.regression_mode:
            mi_scores = mutual_info_regression(X.to_pandas() if hasattr(X, 'to_pandas') else X, 
                                              y.to_numpy() if hasattr(y, 'to_numpy') else y,
                                              discrete_features=False, random_state=42, n_neighbors=5)
        else:
            mi_scores = mutual_info_classif(X.to_pandas() if hasattr(X, 'to_pandas') else X,
                                           y.to_numpy() if hasattr(y, 'to_numpy') else y,
                                           discrete_features=False, random_state=42, n_neighbors=5)
            
        return {feature: float(score) for feature, score in zip(numeric_features, mi_scores)}
        
    def _train_random_forest(self, X, y, numeric_features, use_gpu=False):
        """Train random forest model with GPU if available"""
        log("Training Random Forest model")
        
        if use_gpu and self.use_gpu:
            try:
                if self.regression_mode:
                    model = cuRF_Regressor(n_estimators=100, random_state=42)
                else:
                    model = cuRF_Classifier(n_estimators=100, random_state=42)
                    
                model.fit(X, y)
                importances = model.feature_importances_
                return model, {feature: float(importance) for feature, importance in zip(numeric_features, importances)}
            except Exception as e:
                log(f"GPU Random Forest training failed, falling back to CPU: {e}", LOG_LEVEL_WARNING)
        
        # CPU fallback with multithreading
        if self.regression_mode:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
        model.fit(X.to_pandas() if hasattr(X, 'to_pandas') else X, 
                 y.to_numpy() if hasattr(y, 'to_numpy') else y)
        
        importances = model.feature_importances_
        return model, {feature: float(importance) for feature, importance in zip(numeric_features, importances)}
        
    def _calculate_direction_importance(self, X, y, model, numeric_features):
        """Calculate directional importance for regression models"""
        if not self.regression_mode:
            return {}
            
        log("Calculating directional accuracy correlations")
        
        # Convert to pandas for compatibility
        X_pd = X.to_pandas() if hasattr(X, 'to_pandas') else X
        
        direction_importance = {}
        
        # Process in smaller batches to reduce memory pressure
        batch_size = 200
        num_features = len(numeric_features)
        
        for i in range(0, num_features, batch_size):
            end_idx = min(i + batch_size, num_features)
            current_batch = numeric_features[i:end_idx]
            
            log_progress(i, num_features, "Processing directional importance")
            
            # For each feature, calculate correlation with correct directional prediction
            for j, feature in enumerate(current_batch):
                # Calculate if direction was predicted correctly
                preds = model.predict(X_pd)
                direction_correct = ((self.df['direction'] > 0) == (preds > self.df['Close']))
                
                # Calculate correlation
                feature_dir_corr = np.corrcoef(X_pd[feature], direction_correct)[0, 1]
                direction_importance[feature] = abs(feature_dir_corr)
        
        return direction_importance

    def analyze_feature_importance(self):
        start_section("analyze_feature_importance")
        
        exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close', 
                        'future_close', 'pct_change', 'direction', 'profitable_signal'}
        numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                        if col not in exclude_columns]
        
        log(f"Analyzing importance for {len(numeric_features)} numeric features")
        
        # Prepare data - handle on CPU for consistent preprocessing
        X = self.df[numeric_features].copy()
        y = self.df[self.target_col]
        
        # Handle Inf and NaN values
        log("Preprocessing data - handling missing values")
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        # Clip extreme values
        X = X.clip(lower=np.finfo(np.float64).min / 2, upper=np.finfo(np.float64).max / 2)

        # Ensure y is correctly formatted
        y = y.values.ravel()
        
        # Create GPU versions if using GPU
        if self.use_gpu:
            try:
                log("Moving data to GPU for feature importance calculations")
                X_gpu = cudf.DataFrame.from_pandas(X)
                y_gpu = cudf.Series(y)
                use_gpu = True
            except Exception as e:
                log(f"Failed to move data to GPU: {e}", LOG_LEVEL_WARNING)
                X_gpu = X
                y_gpu = y
                use_gpu = False
        else:
            X_gpu = X
            y_gpu = y
            use_gpu = False

        # Parallel feature importance calculation using ThreadPoolExecutor
        importance_results = {}
        
        try:
            # Run multiple importance methods in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                mi_future = executor.submit(self._calculate_mutual_information, X_gpu, y_gpu, numeric_features, use_gpu)
                rf_future = executor.submit(self._train_random_forest, X_gpu, y_gpu, numeric_features, use_gpu)
                
                # Collect results
                mi_results = mi_future.result()
                rf_model, rf_importance = rf_future.result()
                
                # Direction importance can't be easily parallelized and depends on RF model
                if self.regression_mode:
                    direction_importance = self._calculate_direction_importance(X, y, rf_model, numeric_features)
                else:
                    direction_importance = {}
                
            # Sort and prepare results
            importance_results = {
                'mutual_information': dict(sorted(mi_results.items(), key=lambda x: x[1], reverse=True)),
                'random_forest_importance': dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))
            }
            
            if self.regression_mode:
                importance_results['direction_importance'] = dict(sorted(direction_importance.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            log(f"Error during feature importance calculation: {e}", LOG_LEVEL_ERROR)
            traceback.print_exc()
            importance_results = {'random_forest_importance': {}}
            
        end_section()
        return importance_results
        
    def analyze_shap_values(self, sample_size=100000, max_time_minutes=30, batch_size=500, save_partial=True):
        """
        Calculate SHAP values for feature importance analysis with GPU acceleration when possible
        """
        start_section("analyze_shap_values")
        
        # Setup timing and partial results tracking
        start_time = time.time()
        timeout_seconds = max_time_minutes * 60
        partial_results_file = f"{self.original_target_col}_partial_shap_results.txt"
        
        # Initialize memory tracking
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize partial results file
        if save_partial:
            with open(partial_results_file, 'w') as f:
                f.write(f"SHAP Analysis - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"Target column: {self.target_col}\n")
                f.write(f"Mode: {'Regression' if self.regression_mode else 'Classification'}\n\n")
                f.write(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n\n")
        
        try:
            # Define columns to exclude from analysis
            exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close',
                            'future_close', 'pct_change', 'direction', 'profitable_signal'}
            numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                            if col not in exclude_columns]
            
            log(f"SHAP analysis for {len(numeric_features)} features")
            
            # Prepare data
            X = self.df[numeric_features].copy()
            y = self.df[self.target_col]
            
            # Remove rows with NaN in target
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            log(f"Using {len(X)} valid rows after removing NaN targets")
            
            # Handle problematic values
            log("Preprocessing data for SHAP analysis")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.median(), inplace=True)
            
            # Avoid extreme values that can cause numerical issues
            X = X.clip(lower=np.finfo(np.float32).min / 2, upper=np.finfo(np.float32).max / 2)
            
            # Check for memory availability and adjust sample size if needed
            current_memory = process.memory_info().rss / 1024 / 1024
            available_memory = psutil.virtual_memory().available / 1024 / 1024
            log(f"Current memory usage: {current_memory:.1f}MB, Available: {available_memory:.1f}MB")
            
            # Adjust sample size based on memory - rule of thumb: need ~10x the dataset size in memory
            estimated_memory_per_row = X.memory_usage().sum() / len(X) / 1024 / 1024  # MB per row
            estimated_total_needed = estimated_memory_per_row * sample_size * 10
            
            # Calculate GPU memory if available
            if self.use_gpu:
                try:
                    free_memory, total_memory = cp.cuda.runtime.memGetInfo()
                    available_gpu_memory = free_memory / 1024 / 1024
                    log(f"Available GPU memory: {available_gpu_memory:.1f}MB")
                    
                    # Use the minimum of CPU/GPU memory constraint
                    available_memory = min(available_memory, available_gpu_memory)
                except:
                    log("Could not determine GPU memory availability", LOG_LEVEL_WARNING)
            
            if estimated_total_needed > available_memory * 0.5:  # Use only 50% of available memory
                adjusted_sample_size = int(available_memory * 0.5 / (estimated_memory_per_row * 10))
                log(f"Memory constraint: Adjusting sample size from {sample_size} to {adjusted_sample_size}")
                sample_size = adjusted_sample_size
            
            # Sample data if needed
            if len(X) > sample_size:
                log(f"Sampling {sample_size} rows from {len(X)} for SHAP analysis")
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_indices].copy()  # Explicit copy to free original X from memory
                y_sample = y.iloc[sample_indices].copy()
                
                # Free memory
                del X
                del y
                import gc
                gc.collect()
            else:
                X_sample = X
                y_sample = y
                log(f"Using all {len(X)} rows for SHAP analysis")
            
            # Create an appropriate model based on regression or classification
            if self.use_gpu:
                try:
                    log("Using GPU-accelerated model for SHAP analysis")
                    # Convert to GPU format
                    X_gpu = cudf.DataFrame.from_pandas(X_sample)
                    y_gpu = cudf.Series(y_sample)
                    
                    if self.regression_mode:
                        model = cuRF_Regressor(n_estimators=100, random_state=42)
                    else:
                        model = cuRF_Classifier(n_estimators=100, random_state=42)
                        
                    log("Fitting GPU model")
                    model.fit(X_gpu, y_gpu)
                    
                    # Convert back to CPU for SHAP (which may not fully support cuML models)
                    # Copy model's feature importance for result report purposes
                    gpu_importances = model.feature_importances_
                    
                    # Train equivalent CPU model for SHAP
                    log("Training CPU model for SHAP calculation")
                    if self.regression_mode:
                        cpu_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        cpu_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        
                    cpu_model.fit(X_sample, y_sample)
                    log("CPU model trained for SHAP analysis")
                    
                    # Use CPU model for SHAP
                    model_for_shap = cpu_model
                    X_for_shap = X_sample
                    
                except Exception as e:
                    log(f"GPU model training failed, falling back to CPU: {e}", LOG_LEVEL_WARNING)
                    self.use_gpu = False
                    
                    if self.regression_mode:
                        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        
                    log("Fitting CPU model")
                    model.fit(X_sample, y_sample)
                    model_for_shap = model
                    X_for_shap = X_sample
            else:
                # Use CPU with multithreading
                log("Using CPU model with multithreading for SHAP analysis")
                if self.regression_mode:
                    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    
                log(f"Fitting model with {len(X_sample)} samples and {len(numeric_features)} features")
                fit_start = time.time()
                
                with tqdm(total=1, desc="Training model") as pbar:
                    model.fit(X_sample, y_sample)
                    pbar.update(1)
                    
                fit_time = time.time() - fit_start
                log(f"Model training completed in {fit_time:.2f} seconds")
                model_for_shap = model
                X_for_shap = X_sample
            
            # Check memory again before SHAP calculation
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            log(f"Memory usage after model training: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
            
            # Create the SHAP explainer with timing
            log("Creating SHAP explainer")
            explainer_start = time.time()
            
            try:
                explainer = shap.TreeExplainer(model_for_shap)
                explainer_time = time.time() - explainer_start
                log(f"SHAP explainer created in {explainer_time:.2f} seconds")
            except Exception as e:
                log(f"Error creating SHAP explainer: {e}", LOG_LEVEL_ERROR)
                log("Trying with 'TreeExplainer' parameters adjusted...", LOG_LEVEL_WARNING)
                
                try:
                    # Try with different parameters
                    explainer = shap.TreeExplainer(model_for_shap, feature_perturbation="interventional", model_output="raw")
                    explainer_time = time.time() - explainer_start
                    log(f"Alternative SHAP explainer created in {explainer_time:.2f} seconds")
                except Exception as e2:
                    log(f"Still failed to create explainer: {e2}", LOG_LEVEL_ERROR)
                    raise ValueError("Could not create SHAP explainer with either approach")
            
            # Further reduce sample size for SHAP calculation if needed
            shap_sample_size = min(len(X_for_shap), 10000)  # Cap at 10,000 for SHAP calculation
            if len(X_for_shap) > shap_sample_size:
                log(f"Further sampling to {shap_sample_size} rows for SHAP calculation")
                shap_indices = np.random.choice(len(X_for_shap), shap_sample_size, replace=False)
                X_shap = X_for_shap.iloc[shap_indices].copy()
                # Free memory
                del X_for_shap
                gc.collect()
            else:
                X_shap = X_for_shap
            
            # Use larger batch size for faster processing
            batch_size = min(1000, len(X_shap))
            log(f"Using batch size of {batch_size} for SHAP calculations")
            
            # Calculate SHAP values in batches with detailed progress bar
            log(f"Starting SHAP value calculation on {len(X_shap)} samples in batches of {batch_size}")
            total_rows = len(X_shap)
            num_batches = (total_rows + batch_size - 1) // batch_size
            
            # Prepare for collecting SHAP values
            all_shap_values = None
            processed_rows = 0
            
            with tqdm(total=total_rows, desc="SHAP calculation") as pbar:
                for i in range(0, total_rows, batch_size):
                    # Check for timeout
                    elapsed_time = time.time() - start_time
                    if elapsed_time > timeout_seconds:
                        log(f"SHAP calculation timed out after {elapsed_time/60:.1f} minutes", LOG_LEVEL_WARNING)
                        if processed_rows > 0:
                            log(f"Using partial results from {processed_rows} samples")
                            break
                        else:
                            raise TimeoutError(f"SHAP calculation timed out with no results")
                    
                    # Process current batch
                    batch_start = time.time()
                    end_idx = min(i + batch_size, total_rows)
                    current_batch_size = end_idx - i
                    
                    log(f"Processing SHAP batch {i//batch_size + 1}/{num_batches} ({current_batch_size} samples)", LOG_LEVEL_DEBUG)
                    
                    try:
                        # Calculate SHAP values for this batch
                        batch_X = X_shap.iloc[i:end_idx]
                        batch_shap_values = explainer.shap_values(batch_X)
                        
                        # Initialize storage for all SHAP values on first batch
                        if all_shap_values is None:
                            if isinstance(batch_shap_values, list):
                                # For classification models
                                all_shap_values = [
                                    np.zeros((total_rows,) + arr.shape[1:], dtype=np.float32) 
                                    for arr in batch_shap_values
                                ]
                                for j, arr in enumerate(all_shap_values):
                                    arr[i:end_idx] = batch_shap_values[j]
                            else:
                                # For regression models
                                all_shap_values = np.zeros(
                                    (total_rows,) + batch_shap_values.shape[1:], 
                                    dtype=np.float32
                                )
                                all_shap_values[i:end_idx] = batch_shap_values
                        else:
                            # Add batch values to full array
                            if isinstance(all_shap_values, list):
                                for j, arr in enumerate(all_shap_values):
                                    arr[i:end_idx] = batch_shap_values[j]
                            else:
                                all_shap_values[i:end_idx] = batch_shap_values
                        
                        # Update batch stats
                        batch_time = time.time() - batch_start
                        processed_rows += current_batch_size
                        
                        # Update progress bar
                        pbar.update(current_batch_size)
                        pbar.set_postfix({'time/sample': f"{batch_time/current_batch_size:.4f}s"})
                        
                        # Log batch stats and estimate remaining time
                        avg_time_per_sample = batch_time / current_batch_size
                        remaining_samples = total_rows - processed_rows
                        est_remaining_time = avg_time_per_sample * remaining_samples
                        
                        log(f"Batch {i//batch_size + 1}/{num_batches} completed in {batch_time:.2f}s "
                            f"({batch_time/current_batch_size:.4f}s per sample). "
                            f"Est. remaining: {est_remaining_time/60:.1f} min", LOG_LEVEL_DEBUG)
                        
                        # Save partial results after each batch if requested
                        if save_partial and i > 0 and (i // batch_size) % 5 == 0:
                            partial_shap_values = None
                            
                            # Extract partial results
                            if isinstance(all_shap_values, list):
                                if len(all_shap_values) == 2:  # Binary classification
                                    partial_shap_values = all_shap_values[1][:processed_rows]
                                else:  # Multi-class
                                    partial_shap_values = np.mean(np.abs(np.array([sv[:processed_rows] for sv in all_shap_values])), axis=0)
                            else:  # Regression
                                partial_shap_values = np.abs(all_shap_values[:processed_rows])
                            
                            if partial_shap_values is not None:
                                # Calculate feature importance from partial results
                                mean_abs_shap = np.mean(np.abs(partial_shap_values), axis=0)
                                shap_total = np.sum(mean_abs_shap)
                                mean_abs_shap_normalized = mean_abs_shap / shap_total
                                
                                # Create partial feature importance dictionary
                                partial_importance = {
                                    feature: float(importance)
                                    for feature, importance in zip(numeric_features, mean_abs_shap_normalized)
                                }
                                
                                # Sort and save top features
                                sorted_importance = dict(sorted(partial_importance.items(), key=lambda x: x[1], reverse=True))
                                
                                with open(partial_results_file, 'a') as f:
                                    f.write(f"\n--- Partial SHAP Importance after {processed_rows}/{total_rows} samples "
                                        f"({processed_rows/total_rows:.1%}) - {datetime.now().strftime('%H:%M:%S')} ---\n")
                                    for i, (feature, score) in enumerate(list(sorted_importance.items())[:30]):
                                        f.write(f"{i+1}. {feature}: {score:.6f}\n")
                    
                    except Exception as e:
                        log(f"Error in SHAP batch {i//batch_size + 1}: {str(e)}", LOG_LEVEL_ERROR)
                        log(f"Batch traceback: {traceback.format_exc()}", LOG_LEVEL_DEBUG)
                        
                        if processed_rows > 0:
                            log(f"Using partial results from {processed_rows} samples")
                            break
                        else:
                            raise ValueError(f"SHAP calculation failed in first batch: {str(e)}")
            
            # Check if we got any results
            if all_shap_values is None or processed_rows == 0:
                log("No SHAP values were calculated successfully", LOG_LEVEL_ERROR)
                return {'shap_importance': {}}
            
            # Process SHAP values based on type
            log(f"Processing final SHAP results from {processed_rows} samples")
            
            # If we didn't process all rows, trim the arrays
            if processed_rows < total_rows:
                if isinstance(all_shap_values, list):
                    all_shap_values = [sv[:processed_rows] for sv in all_shap_values]
                else:
                    all_shap_values = all_shap_values[:processed_rows]
            
            # Process based on model type
            if isinstance(all_shap_values, list):
                # Classification model
                if len(all_shap_values) == 2:  # Binary classification
                    shap_values = all_shap_values[1]  # Class 1 importance
                else:  # Multi-class
                    shap_values = np.mean(np.abs(np.array(all_shap_values)), axis=0)
            elif isinstance(all_shap_values, np.ndarray) and all_shap_values.ndim > 2:
                # Multi-output regression
                shap_values = np.mean(np.abs(all_shap_values), axis=2)
            else:
                # Single output regression
                shap_values = np.abs(all_shap_values)
            
            # Calculate mean absolute SHAP values per feature
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Normalize to sum to 1
            shap_total = np.sum(mean_abs_shap)
            mean_abs_shap_normalized = mean_abs_shap / shap_total
            
            # Create feature:importance dictionary
            shap_importance = {
                feature: float(importance)
                for feature, importance in zip(numeric_features, mean_abs_shap_normalized)
            }
            
            # Sort and return results
            sorted_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Write final results to partial file
            if save_partial:
                with open(partial_results_file, 'a') as f:
                    f.write(f"\n\n{'='*20} FINAL SHAP RESULTS {'='*20}\n")
                    f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"GPU Acceleration: {'Enabled' if self.use_gpu else 'Disabled'}\n")
                    f.write(f"Processed {processed_rows} samples out of {total_rows} total\n\n")
                    
                    f.write("Top 50 features by SHAP importance:\n")
                    for i, (feature, score) in enumerate(list(sorted_importance.items())[:50]):
                        f.write(f"{i+1}. {feature}: {score:.6f}\n")
            
            log(f"SHAP analysis complete - processed {processed_rows} samples")
            end_section()
            
            return {'shap_importance': sorted_importance}
            
        except Exception as e:
            log(f"Error during SHAP analysis: {e}", LOG_LEVEL_ERROR)
            log(traceback.format_exc(), LOG_LEVEL_DEBUG)
            
            # Try to save whatever we've calculated so far
            if 'all_shap_values' in locals() and all_shap_values is not None and processed_rows > 0 and save_partial:
                log(f"Saving partial SHAP results from {processed_rows} samples before exiting")
                
                try:
                    # Similar calculation as in the batch processing loop
                    partial_shap_values = None
                    
                    if isinstance(all_shap_values, list):
                        if len(all_shap_values) == 2:
                            partial_shap_values = all_shap_values[1][:processed_rows]
                        else:
                            partial_shap_values = np.mean(np.abs(np.array([sv[:processed_rows] for sv in all_shap_values])), axis=0)
                    else:
                        partial_shap_values = np.abs(all_shap_values[:processed_rows])
                    
                    if partial_shap_values is not None:
                        mean_abs_shap = np.mean(np.abs(partial_shap_values), axis=0)
                        shap_total = np.sum(mean_abs_shap)
                        mean_abs_shap_normalized = mean_abs_shap / shap_total
                        
                        partial_importance = {
                            feature: float(importance)
                            for feature, importance in zip(numeric_features, mean_abs_shap_normalized)
                        }
                        
                        sorted_importance = dict(sorted(partial_importance.items(), key=lambda x: x[1], reverse=True))
                        
                        with open(partial_results_file, 'a') as f:
                            f.write(f"\n\n{'='*20} ERROR RECOVERY - PARTIAL RESULTS {'='*20}\n")
                            f.write(f"Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"Error message: {str(e)}\n")
                            f.write(f"Processed {processed_rows} samples before error\n\n")
                            
                            f.write("Top 30 features from partial results:\n")
                            for i, (feature, score) in enumerate(list(sorted_importance.items())[:30]):
                                f.write(f"{i+1}. {feature}: {score:.6f}\n")
                        
                        end_section()
                        return {'shap_importance': sorted_importance}
                
                except Exception as inner_e:
                    log(f"Failed to save partial results: {inner_e}", LOG_LEVEL_ERROR)
            
            end_section()
            return {'shap_importance': {}}

    def analyze_time_series_stability(self, n_splits=5):
        start_section("analyze_time_series_stability")
        log("Analyzing time series stability with TimeSeriesSplit")
        
        exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close',
                         'future_close', 'pct_change', 'direction', 'profitable_signal'}
        numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_columns]
        
        X = self.df[numeric_features].copy()
        y = self.df[self.target_col]
        
        # Handle data issues
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        X = X.clip(lower=np.finfo(np.float64).min / 2, upper=np.finfo(np.float64).max / 2)
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        importance_over_time = {feature: [] for feature in numeric_features}
        
        # Use multithreading for model training in time series splits
        def process_split(split_data):
            split_index, (train_idx, test_idx) = split_data
            log(f"Processing time split {split_index+1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model with multithreading
            if self.regression_mode:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
            model.fit(X_train, y_train)
            
            # Return importances for this time period
            return dict(zip(numeric_features, model.feature_importances_))
            
        # Process splits in parallel
        with ThreadPoolExecutor(max_workers=min(n_splits, os.cpu_count())) as executor:
            split_results = list(executor.map(
                process_split, 
                enumerate(tscv.split(X))
            ))
        
        # Combine results from all splits
        for split_importances in split_results:
            for feature, importance in split_importances.items():
                importance_over_time[feature].append(importance)
        
        # Calculate stability metrics
        stability_metrics = {}
        for feature, importances in importance_over_time.items():
            try:
                cv = np.std(importances) / np.mean(importances) if np.mean(importances) > 0 else float('inf')
                trend = np.polyfit(range(len(importances)), importances, 1)[0]
                
                stability_metrics[feature] = {
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'coefficient_of_variation': cv,
                    'trend': trend  # Slope of importance over time
                }
            except:
                # Handle any calculation errors
                stability_metrics[feature] = {
                    'mean_importance': 0.0,
                    'std_importance': 0.0,
                    'coefficient_of_variation': float('inf'),
                    'trend': 0.0
                }
        
        # Sort by mean importance
        sorted_stability = dict(sorted(stability_metrics.items(), 
                                     key=lambda x: x[1]['mean_importance'], 
                                     reverse=True))
        
        end_section()
        return sorted_stability

def analyze_trading_dataset(file_path, target_col=target_col, regression_mode=regression_mode, 
                         forecast_periods=14, run_shap=True, max_shap_time_minutes=30):
    """
    Analyze trading dataset with incremental file writing
    """
    start_section("analyze_trading_dataset")
    
    # Set up output file path
    output_file = f"{target_col}_analysis_results.txt"
    # Start with a new file
    write_partial_results({}, "Analysis Started", file_path=output_file, mode='w')
    
    log(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    log(f"Dataset loaded with shape: {df.shape}")
    
    log("Initializing analyzer")
    analyzer = EnhancedTradingDataAnalyzer(df, target_col, regression_mode, forecast_periods)
    
    # Initialize results dictionary
    results = {}
    
    try:
        # Run certain analyses in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit parallel tasks
            structure_future = executor.submit(analyzer.analyze_dataset_structure)
            stats_future = executor.submit(analyzer.analyze_feature_statistics)
            patterns_future = executor.submit(analyzer.analyze_periodic_patterns)
            
            # Collect results as they complete
            log("Waiting for parallel analyses to complete...")
            
            # Dataset structure
            results['dataset_structure'] = structure_future.result()
            write_partial_results(results, "Dataset Structure", file_path=output_file)
            
            # Feature statistics
            results['feature_statistics'] = stats_future.result()
            write_partial_results(results, "Feature Statistics", file_path=output_file)
            
            # Periodic patterns
            results['periodic_patterns'] = patterns_future.result()
            write_partial_results(results, "Periodic Patterns", file_path=output_file)
    
    except Exception as e:
        log(f"Error during parallel analyses: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Analyze feature importance
        log("Analyzing feature importance")
        results['feature_importance'] = analyzer.analyze_feature_importance()
        write_partial_results(results, "Feature Importance", file_path=output_file)
    except Exception as e:
        log(f"Error during feature importance analysis: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Analyze SHAP values
        if run_shap:
            log("Starting SHAP analysis")
            results['shap_analysis'] = analyzer.analyze_shap_values(max_time_minutes=max_shap_time_minutes)
        else:
            log("Skipping SHAP analysis")
            results['shap_analysis'] = {'shap_importance': {}}
        
        write_partial_results(results, "SHAP Analysis", file_path=output_file)
    except Exception as e:
        log(f"Error during SHAP analysis: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
        results['shap_analysis'] = {'shap_importance': {}}
        write_partial_results(results, "SHAP Analysis (Failed)", file_path=output_file)
    
    try:
        # Analyze time series stability
        log("Analyzing time series stability")
        results['time_series_stability'] = analyzer.analyze_time_series_stability(n_splits=5)
        write_partial_results(results, "Time Series Stability", file_path=output_file)
    except Exception as e:
        log(f"Error during time series stability analysis: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Write important features summary
        log("Writing important features")
        write_partial_results(results, "Important Features", file_path=output_file)
        
        # Also write to separate file
        with open(f'{target_col}_important_features.txt', 'w') as f:
            f.write("=== IMPORTANT FEATURES ===\n\n")
            f.write("# Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            
            # Calculate important features - simplified version
            important_features = set()
            for method_name in ['mutual_information', 'random_forest_importance']:
                if method_name in results.get('feature_importance', {}):
                    method_dict = results['feature_importance'][method_name]
                    top_features = sorted(method_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                    for feature, _ in top_features:
                        important_features.add(feature)
            
            if 'shap_importance' in results.get('shap_analysis', {}):
                top_shap = sorted(results['shap_analysis']['shap_importance'].items(), 
                                 key=lambda x: x[1], reverse=True)[:20]
                for feature, _ in top_shap:
                    important_features.add(feature)
            
            # Write to file
            for feature in sorted(important_features):
                f.write(feature + '\n')
    except Exception as e:
        log(f"Error writing important features: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    # Write completion message
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*30} ANALYSIS COMPLETED {'='*30}\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    end_section()
    return results, df

def run_configuration(config, file_path, run_shap, max_shap_time):
    """Function to run a single configuration"""
    target = config["target_col"]
    regression = config["regression_mode"]
    
    log(f"Starting analysis for target_col={target}, regression_mode={regression}")
    
    try:
        results, df = analyze_trading_dataset(
            file_path, 
            target_col=target,
            regression_mode=regression,
            run_shap=run_shap, 
            max_shap_time_minutes=max_shap_time
        )
        log(f"Analysis for {target} completed successfully")
        return True
    except Exception as e:
        log(f"Fatal error during analysis for {target}: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # File Path should be csv of all features
    file_path = 'EURUSD_1min_sampled_features.csv'
    
    # Set to False to skip SHAP if it's causing issues
    run_shap = True
    max_shap_time = 30  # minutes
    
    start_section("main")
    
    # Configuration to run
    configurations = [
        {"target_col": "long_signal", "regression_mode": False},
        {"target_col": "short_signal", "regression_mode": False},
        {"target_col": "Close", "regression_mode": True}
    ]
    
    # Run all configurations in parallel or sequentially
    cores_available = os.cpu_count()
    log(f"Detected {cores_available} CPU cores")
    
    if cores_available >= 6:  # Only parallelize with enough cores
        # Run in parallel with ProcessPoolExecutor
        log(f"Running {len(configurations)} configurations in parallel")
        with ProcessPoolExecutor(max_workers=len(configurations)) as executor:
            futures = []
            for config in configurations:
                futures.append(executor.submit(
                    run_configuration, config, file_path, run_shap, max_shap_time
                ))
            
            # Wait for all configurations to complete
            for i, future in enumerate(futures):
                config = configurations[i]
                try:
                    result = future.result()
                    status = "successful" if result else "failed"
                    log(f"Configuration {config['target_col']} completed with status: {status}")
                except Exception as e:
                    log(f"Exception in configuration {config['target_col']}: {e}")
    else:
        # Run sequentially
        log("Running configurations sequentially")
        for config in configurations:
            run_configuration(config, file_path, run_shap, max_shap_time)
    
    print_timing_summary()
    end_section()
    
    log("All script executions completed")
# %%