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
import pickle
import joblib
from datetime import datetime
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# Enhanced Feature Selection Imports
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, 
    RFE, RFECV, SelectFromModel,
    mutual_info_regression, mutual_info_classif
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV
import warnings
warnings.filterwarnings('ignore')

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
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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

class AdvancedFeatureSelector:
    """
    Advanced feature selection combining multiple methods with validation
    """
    
    def __init__(self, regression_mode=True, target_features=75, 
                 validation_splits=5, random_state=42):
        self.regression_mode = regression_mode
        self.target_features = target_features
        self.validation_splits = validation_splits
        self.random_state = random_state
        self.selection_results = {}
        
    def prepare_data(self, df, feature_cols, target_col):
        """Prepare and clean data for feature selection"""
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values and infinities
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        
        # Remove rows with missing targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        return X, y
    
    def method_1_statistical_selection(self, X, y):
        """Method 1: Statistical-based selection (Mutual Information + Percentile)"""
        print("Method 1: Statistical Selection")
        
        if self.regression_mode:
            selector = SelectPercentile(
                score_func=mutual_info_regression, 
                percentile=min(80, (self.target_features * 100) // len(X.columns))
            )
        else:
            selector = SelectPercentile(
                score_func=mutual_info_classif,
                percentile=min(80, (self.target_features * 100) // len(X.columns))
            )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        scores = dict(zip(X.columns, selector.scores_))
        
        self.selection_results['statistical'] = {
            'features': selected_features[:self.target_features],
            'scores': scores,
            'method': 'Mutual Information + Percentile'
        }
        
        return selected_features[:self.target_features]
    
    def method_2_recursive_elimination(self, X, y):
        """Method 2: Recursive Feature Elimination with Cross-Validation"""
        print("Method 2: Recursive Feature Elimination")
        
        if self.regression_mode:
            estimator = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        
        # Use RFECV for automatic feature number selection, but limit to target
        cv = TimeSeriesSplit(n_splits=3)  # Faster for initial selection
        
        selector = RFECV(
            estimator=estimator,
            step=5,  # Remove 5 features at a time for speed
            cv=cv,
            scoring='neg_mean_squared_error' if self.regression_mode else 'accuracy',
            n_jobs=-1,
            min_features_to_select=min(10, self.target_features // 2)
        )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # If we got more than target, take top by ranking
        if len(selected_features) > self.target_features:
            feature_ranking = dict(zip(X.columns, selector.ranking_))
            selected_features = sorted(selected_features, 
                                     key=lambda x: feature_ranking[x])[:self.target_features]
        
        self.selection_results['rfe'] = {
            'features': selected_features,
            'scores': dict(zip(X.columns, selector.ranking_)),
            'method': 'Recursive Feature Elimination CV'
        }
        
        return selected_features
    
    def method_3_regularization_based(self, X, y):
        """Method 3: Regularization-based selection (Lasso/L1)"""
        print("Method 3: Regularization-based Selection")
        
        if self.regression_mode:
            # Use LassoCV for automatic alpha selection
            selector = LassoCV(cv=3, random_state=self.random_state, n_jobs=-1)
        else:
            # Use L1 Logistic Regression
            selector = LogisticRegressionCV(
                cv=3, penalty='l1', solver='liblinear', 
                random_state=self.random_state, n_jobs=-1
            )
        
        selector.fit(X, y)
        
        # Get feature coefficients
        if hasattr(selector, 'coef_'):
            if self.regression_mode:
                coefficients = np.abs(selector.coef_)
            else:
                coefficients = np.abs(selector.coef_[0])
        else:
            coefficients = np.zeros(len(X.columns))
        
        # Select features with non-zero coefficients, then top by magnitude
        non_zero_idx = coefficients > 1e-6
        feature_scores = dict(zip(X.columns, coefficients))
        
        selected_features = [feat for feat, coef in feature_scores.items() if coef > 1e-6]
        selected_features = sorted(selected_features, 
                                 key=lambda x: feature_scores[x], reverse=True)[:self.target_features]
        
        self.selection_results['regularization'] = {
            'features': selected_features,
            'scores': feature_scores,
            'method': 'Lasso/L1 Regularization'
        }
        
        return selected_features
    
    def method_4_model_based_selection(self, X, y):
        """Method 4: Model-based selection with multiple algorithms"""
        print("Method 4: Model-based Selection")
        
        if self.regression_mode:
            estimator = RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        estimator.fit(X, y)
        
        # Select features based on importance threshold
        importances = estimator.feature_importances_
        feature_scores = dict(zip(X.columns, importances))
        
        # Take top features by importance
        selected_features = sorted(X.columns, key=lambda x: feature_scores[x], reverse=True)[:self.target_features]
        
        self.selection_results['model_based'] = {
            'features': selected_features,
            'scores': feature_scores,
            'method': 'Random Forest Importance'
        }
        
        return selected_features
    
    def method_5_ensemble_ranking(self, X, y):
        """Method 5: Ensemble ranking combining multiple methods"""
        print("Method 5: Ensemble Ranking")
        
        # Run all individual methods
        stat_features = self.method_1_statistical_selection(X, y)
        rfe_features = self.method_2_recursive_elimination(X, y)
        reg_features = self.method_3_regularization_based(X, y)
        model_features = self.method_4_model_based_selection(X, y)
        
        # Count votes for each feature
        all_features = set(X.columns)
        feature_votes = {feat: 0 for feat in all_features}
        feature_scores_sum = {feat: 0.0 for feat in all_features}
        
        # Weight each method equally and normalize scores
        methods = ['statistical', 'rfe', 'model_based', 'regularization']
        for method in methods:
            if method in self.selection_results:
                selected = self.selection_results[method]['features']
                scores = self.selection_results[method]['scores']
                
                # Normalize scores to 0-1 range
                score_values = list(scores.values())
                if len(score_values) > 0:
                    min_score = min(score_values)
                    max_score = max(score_values)
                    score_range = max_score - min_score
                    
                    if score_range > 0:
                        for feat in selected:
                            feature_votes[feat] += 1
                            normalized_score = (scores[feat] - min_score) / score_range
                            feature_scores_sum[feat] += normalized_score
                    else:
                        for feat in selected:
                            feature_votes[feat] += 1
                            feature_scores_sum[feat] += 1.0
        
        # Combine votes and normalized scores
        ensemble_scores = {}
        for feat in all_features:
            # Weighted combination: 60% votes, 40% average score
            vote_score = feature_votes[feat] / len(methods)
            avg_score = feature_scores_sum[feat] / max(1, feature_votes[feat])
            ensemble_scores[feat] = 0.6 * vote_score + 0.4 * avg_score
        
        # Select top features
        selected_features = sorted(all_features, key=lambda x: ensemble_scores[x], reverse=True)[:self.target_features]
        
        self.selection_results['ensemble'] = {
            'features': selected_features,
            'scores': ensemble_scores,
            'votes': feature_votes,
            'method': 'Ensemble Ranking'
        }
        
        return selected_features
    
    def validate_selection(self, X, y, selected_features, method_name):
        """Validate feature selection using cross-validation"""
        print(f"Validating {method_name} selection...")
        
        if len(selected_features) == 0:
            return {'cv_score': 0.0, 'std_score': 0.0}
        
        X_selected = X[selected_features]
        
        if self.regression_mode:
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            scoring = 'neg_mean_squared_error'
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=self.random_state, n_jobs=-1)
            scoring = 'accuracy'
        
        cv = TimeSeriesSplit(n_splits=self.validation_splits)
        scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        return {
            'cv_score': np.mean(scores),
            'std_score': np.std(scores),
            'feature_count': len(selected_features)
        }
    
    def select_best_features(self, df, feature_cols, target_col, validate=True):
        """Main method to select the best features using all methods"""
        print(f"Starting advanced feature selection for {len(feature_cols)} features -> {self.target_features} target features")
        
        X, y = self.prepare_data(df, feature_cols, target_col)
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Run all methods
        results = {}
        
        # Method 5 automatically runs 1-4
        ensemble_features = self.method_5_ensemble_ranking(X, y)
        
        # Validate each method if requested
        if validate:
            print("\nValidating feature selections...")
            for method_name, method_data in self.selection_results.items():
                validation_results = self.validate_selection(X, y, method_data['features'], method_name)
                self.selection_results[method_name]['validation'] = validation_results
        
        # Find best method based on validation score
        best_method = 'ensemble'  # default
        best_score = float('-inf')
        
        if validate:
            for method_name, method_data in self.selection_results.items():
                if 'validation' in method_data:
                    score = method_data['validation']['cv_score']
                    if score > best_score:
                        best_score = score
                        best_method = method_name
        
        print(f"\nBest performing method: {best_method}")
        
        return {
            'best_features': self.selection_results[best_method]['features'],
            'best_method': best_method,
            'all_results': self.selection_results,
            'summary': self.get_selection_summary()
        }
    
    def get_selection_summary(self):
        """Get a summary of all selection methods"""
        summary = {}
        
        for method_name, method_data in self.selection_results.items():
            summary[method_name] = {
                'feature_count': len(method_data['features']),
                'method_description': method_data['method']
            }
            
            if 'validation' in method_data:
                summary[method_name]['cv_score'] = method_data['validation']['cv_score']
                summary[method_name]['score_std'] = method_data['validation']['std_score']
        
        return summary

def write_partial_results(results, section_name, file_path, mode='a'):  # Remove default None
    """
    Write partial results to file as they become available
    """
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
        
        elif section_name == "Feature Filtering":
            if 'feature_filtering' in results:
                filtering_info = results['feature_filtering']
                f.write(f"Original features: {filtering_info['original_count']}\n")
                f.write(f"Features after constant removal: {filtering_info['after_constant_removal']}\n")
                f.write(f"Features after collinearity removal: {filtering_info['after_collinearity_removal']}\n")
                f.write(f"Features after stability filtering: {filtering_info['final_count']}\n\n")
                
                f.write("Removed features by category:\n")
                f.write(f"Constant features ({len(filtering_info['removed_constant'])}): {', '.join(filtering_info['removed_constant'][:10])}\n")
                if len(filtering_info['removed_constant']) > 10:
                    f.write(f"... and {len(filtering_info['removed_constant']) - 10} more\n")
                    
                f.write(f"High collinearity features ({len(filtering_info['removed_collinear'])}): {', '.join(filtering_info['removed_collinear'][:10])}\n")
                if len(filtering_info['removed_collinear']) > 10:
                    f.write(f"... and {len(filtering_info['removed_collinear']) - 10} more\n")
                    
                f.write(f"Unstable features ({len(filtering_info['removed_unstable'])}): {', '.join(filtering_info['removed_unstable'][:10])}\n")
                if len(filtering_info['removed_unstable']) > 10:
                    f.write(f"... and {len(filtering_info['removed_unstable']) - 10} more\n")
                    
                f.write(f"\nFiltered feature set ({len(filtering_info['final_features'])}): {', '.join(filtering_info['final_features'])}\n")

        elif section_name == "Enhanced Feature Selection":
            if 'enhanced_selection' in results:
                selection_info = results['enhanced_selection']
                f.write(f"Enhanced selection method: {selection_info['best_method']}\n")
                f.write(f"Final selected features: {len(selection_info['best_features'])}\n\n")
                
                # Write method comparison
                f.write("Method Performance Comparison:\n" + "-" * 40 + "\n")
                for method, summary in selection_info['summary'].items():
                    f.write(f"{method.upper()}:\n")
                    f.write(f"  Method: {summary['method_description']}\n")
                    f.write(f"  Features: {summary['feature_count']}\n")
                    if 'cv_score' in summary:
                        f.write(f"  CV Score: {summary['cv_score']:.4f} ± {summary['score_std']:.4f}\n")
                    f.write("\n")
                
                # Write selected features
                f.write(f"Selected Features ({len(selection_info['best_features'])}):\n")
                for i, feature in enumerate(selection_info['best_features'], 1):
                    f.write(f"{i:2d}. {feature}\n")
        
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
            # Use enhanced selection results if available
            if 'enhanced_selection' in results:
                f.write("=== ENHANCED SELECTED FEATURES ===\n\n")
                f.write(f"Selection method: {results['enhanced_selection']['best_method']}\n")
                f.write(f"Total features: {len(results['enhanced_selection']['best_features'])}\n\n")
                
                f.write("# Top features selected by enhanced method:\n")
                for i, feature in enumerate(results['enhanced_selection']['best_features'], 1):
                    f.write(f"{i:2d}. {feature}\n")
            else:
                # Fallback to original logic
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

                f.write("=== IMPORTANT FEATURES (Legacy Method) ===\n\n")
                f.write("# Features with high importance and stability:\n")
                for feature in sorted(important_features):
                    f.write(feature + '\n')
                
                f.write("\n# Top stable features across time periods:\n")
                for feature, importance in stable_features[:20]:
                    f.write(f"{feature}: {importance:.4f}\n")
        
        elif section_name == "Model Information":
            if 'saved_models' in results:
                f.write("Saved Models:\n")
                for model_name, model_path in results['saved_models'].items():
                    f.write(f"  {model_name}: {model_path}\n")
        
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
    
    # Add GPU memory tracking if available - with error handling
    if GPU_AVAILABLE:
        try:
            # Check if CUDA is properly initialized first
            cp.cuda.runtime.getDeviceCount()  # This will throw if CUDA not available
            free_memory, total_memory = cp.cuda.runtime.memGetInfo()
            gpu_memory_used = (total_memory - free_memory) / 1024 / 1024
            gpu_utilization = 100 * (1 - free_memory / total_memory)
            original_print(f"    GPU Memory: {gpu_memory_used:.1f}MB ({gpu_utilization:.1f}% used)")
        except:
            # Silently disable GPU tracking if there's any issue
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

def calculate_feature_correlation_matrix(df, numeric_features, correlation_threshold=0.95):
    """
    Calculate correlation matrix and identify highly correlated features to remove
    """
    log(f"Calculating correlation matrix for {len(numeric_features)} features")
    
    # Calculate correlation matrix in chunks to handle memory
    chunk_size = 100
    correlation_matrix = np.eye(len(numeric_features))
    
    for i in range(0, len(numeric_features), chunk_size):
        end_i = min(i + chunk_size, len(numeric_features))
        for j in range(i, len(numeric_features), chunk_size):
            end_j = min(j + chunk_size, len(numeric_features))
            
            # Calculate correlation for this chunk
            chunk_data_i = df[numeric_features[i:end_i]]
            chunk_data_j = df[numeric_features[j:end_j]]
            
            chunk_corr = np.corrcoef(chunk_data_i.T, chunk_data_j.T)
            
            if i == j:  # Diagonal chunk
                correlation_matrix[i:end_i, j:end_j] = chunk_corr[:end_i-i, :end_j-j]
            else:
                correlation_matrix[i:end_i, j:end_j] = chunk_corr[:end_i-i, end_i-i:end_i-i+end_j-j]
                correlation_matrix[j:end_j, i:end_i] = chunk_corr[end_i-i:end_i-i+end_j-j, :end_i-i]
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(numeric_features)):
        for j in range(i+1, len(numeric_features)):
            if abs(correlation_matrix[i, j]) > correlation_threshold:
                high_corr_pairs.append((i, j, correlation_matrix[i, j]))
    
    log(f"Found {len(high_corr_pairs)} highly correlated pairs (threshold: {correlation_threshold})")
    
    # Decide which features to remove - keep the one with higher variance
    features_to_remove = set()
    for i, j, corr in high_corr_pairs:
        feature_i = numeric_features[i]
        feature_j = numeric_features[j]
        
        var_i = df[feature_i].var()
        var_j = df[feature_j].var()
        
        # Remove the feature with lower variance
        if var_i < var_j:
            features_to_remove.add(feature_i)
        else:
            features_to_remove.add(feature_j)
    
    return list(features_to_remove), high_corr_pairs

def quick_stability_check(df, numeric_features, target_col, n_splits=3, cv_threshold=1.0):
    """
    Quick stability check using smaller splits to identify unstable features early
    """
    log(f"Quick stability check for {len(numeric_features)} features")
    
    # Use only a subset for quick check
    sample_size = min(50000, len(df))
    if len(df) > sample_size:
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        df_sample = df.iloc[sample_indices]
    else:
        df_sample = df
    
    X = df_sample[numeric_features].copy()
    y = df_sample[target_col]
    
    # Handle data issues
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    importance_over_time = {feature: [] for feature in numeric_features}
    
    for train_idx, test_idx in tscv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # Use a smaller, faster model for quick check
        if len(np.unique(y_train)) > 2:  # Regression
            model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        else:  # Classification
            model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
            
        model.fit(X_train, y_train)
        
        for feature, importance in zip(numeric_features, model.feature_importances_):
            importance_over_time[feature].append(importance)
    
    # Calculate coefficient of variation for each feature
    unstable_features = []
    for feature, importances in importance_over_time.items():
        if len(importances) > 1:
            cv = np.std(importances) / (np.mean(importances) + 1e-8)
            if cv > cv_threshold:
                unstable_features.append(feature)
    
    log(f"Identified {len(unstable_features)} unstable features (CV > {cv_threshold})")
    return unstable_features

class EnhancedTradingDataAnalyzer:
    def __init__(self, df, target_col='long_signal', regression_mode=False, forecast_periods=14):  # Use string defaults
        self.df = df.copy()
        self.original_target_col = target_col  # Store the actual target column
        self.regression_mode = regression_mode
        self.forecast_periods = forecast_periods
        self.feature_stats = {}
        self.saved_models = {}
        
        # Create models directory using the INSTANCE target_col, not global
        self.models_dir = f"models_{self.original_target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 🔥 OPTIMIZED GPU INITIALIZATION
        self.use_gpu = False
        self.gpu_df = None
        
        if GPU_AVAILABLE:
            try:
                # More thorough GPU test
                log("Testing GPU functionality...")
                
                # Test CuPy
                test_array = cp.array([1, 2, 3, 4, 5])
                test_result = cp.sum(test_array)
                
                # Test cuDF
                test_df = cudf.DataFrame({'test': [1, 2, 3]})
                
                # Test cuML with a tiny model
                from cuml.ensemble import RandomForestClassifier as cuRF_Test
                test_X = cp.array([[1, 2], [3, 4], [5, 6]])
                test_y = cp.array([0, 1, 0])
                test_model = cuRF_Test(n_estimators=2)
                test_model.fit(test_X, test_y)
                
                # If we get here, GPU is fully functional
                self.use_gpu = True
                log("✅ GPU test successful - All RAPIDS libraries working")
                
                # Get GPU memory info
                free_memory, total_memory = cp.cuda.runtime.memGetInfo()
                gpu_memory_gb = total_memory / (1024**3)
                free_memory_gb = free_memory / (1024**3)
                log(f"GPU Memory: {free_memory_gb:.1f}GB free / {gpu_memory_gb:.1f}GB total")
                
            except Exception as e:
                log(f"GPU initialization failed: {e}", LOG_LEVEL_WARNING)
                log("Falling back to CPU-only mode")
                self.use_gpu = False
        else:
            log("RAPIDS not available - using CPU-only mode")
        
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
        
        # Apply feature filtering
        self.feature_filtering_results = self._filter_features()
        
        # Convert to GPU dataframe if GPU available and enabled
        if self.use_gpu:
            try:
                log("Converting dataframe to GPU memory")
                self.gpu_df = cudf.DataFrame.from_pandas(self.df)
                log(f"DataFrame successfully moved to GPU memory")
            except Exception as e:
                log(f"Failed to move dataframe to GPU: {e}", LOG_LEVEL_WARNING)
                self.use_gpu = False
                self.gpu_df = None

    def run_enhanced_feature_selection(self, target_features=75, validate=True):
        """
        Run enhanced feature selection using AdvancedFeatureSelector
        
        Args:
            target_features: Number of features to select
            validate: Whether to validate selections with cross-validation
        
        Returns:
            Dictionary with selection results
        """
        start_section("run_enhanced_feature_selection")
        
        log(f"Starting enhanced feature selection to select {target_features} features")
        
        # Get filtered features from basic filtering
        filtered_features = self.feature_filtering_results['final_features']
        log(f"Running enhanced selection on {len(filtered_features)} pre-filtered features")
        
        # Initialize the advanced selector
        selector = AdvancedFeatureSelector(
            regression_mode=self.regression_mode,
            target_features=target_features,
            validation_splits=5,
            random_state=42
        )
        
        try:
            # Run the enhanced selection
            selection_results = selector.select_best_features(
                df=self.df,
                feature_cols=filtered_features,
                target_col=self.target_col,
                validate=validate
            )
            
            log(f"Enhanced selection completed. Best method: {selection_results['best_method']}")
            log(f"Selected {len(selection_results['best_features'])} features")
            
            # Save detailed results to file
            detailed_file = f"{self.original_target_col}_enhanced_selection_details.txt"
            with open(detailed_file, 'w') as f:
                f.write("=== ENHANCED FEATURE SELECTION DETAILS ===\n\n")
                f.write(f"Target: {self.original_target_col}\n")
                f.write(f"Mode: {'Regression' if self.regression_mode else 'Classification'}\n")
                f.write(f"Target Features: {target_features}\n")
                f.write(f"Best Method: {selection_results['best_method']}\n\n")
                
                f.write("METHOD COMPARISON:\n" + "="*50 + "\n")
                for method, summary in selection_results['summary'].items():
                    f.write(f"\n{method.upper()}:\n")
                    f.write(f"  Description: {summary['method_description']}\n")
                    f.write(f"  Features Selected: {summary['feature_count']}\n")
                    if 'cv_score' in summary:
                        f.write(f"  Cross-validation Score: {summary['cv_score']:.6f} ± {summary['score_std']:.6f}\n")
                
                f.write(f"\n\nSELECTED FEATURES ({len(selection_results['best_features'])}):\n")
                f.write("="*50 + "\n")
                for i, feature in enumerate(selection_results['best_features'], 1):
                    f.write(f"{i:2d}. {feature}\n")
                
                # Write feature scores for best method
                if 'all_results' in selection_results:
                    best_method_data = selection_results['all_results'][selection_results['best_method']]
                    if 'scores' in best_method_data:
                        f.write(f"\n\nFEATURE SCORES ({selection_results['best_method'].upper()}):\n")
                        f.write("="*50 + "\n")
                        
                        # Sort scores for selected features
                        selected_scores = {
                            feat: best_method_data['scores'][feat] 
                            for feat in selection_results['best_features']
                            if feat in best_method_data['scores']
                        }
                        
                        for i, (feature, score) in enumerate(
                            sorted(selected_scores.items(), key=lambda x: x[1], reverse=True), 1
                        ):
                            f.write(f"{i:2d}. {feature}: {score:.6f}\n")
            
            log(f"Detailed selection results saved to: {detailed_file}")
            
            end_section()
            return selection_results
            
        except Exception as e:
            log(f"Enhanced feature selection failed: {e}", LOG_LEVEL_ERROR)
            traceback.print_exc()
            
            # Return fallback selection (top features by basic importance)
            log("Falling back to basic feature selection")
            fallback_features = filtered_features[:target_features]
            
            end_section()
            return {
                'best_features': fallback_features,
                'best_method': 'fallback',
                'all_results': {},
                'summary': {'fallback': {'method_description': 'Basic top features fallback', 'feature_count': len(fallback_features)}}
            }

    def _filter_features(self, correlation_threshold=0.95, cv_threshold=1.0):
        """
        Comprehensive feature filtering to remove problematic features
        """
        start_section("_filter_features")
        
        # Get initial numeric features
        exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close', 
                          'future_close', 'pct_change', 'direction', 'profitable_signal'}
        numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude_columns]
        
        original_count = len(numeric_features)
        log(f"Starting with {original_count} numeric features")
        
        # Step 1: Remove constant features
        log("Step 1: Removing constant features")
        constant_features = self._remove_constant_features(numeric_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
        after_constant = len(numeric_features)
        log(f"Removed {len(constant_features)} constant features, {after_constant} remaining")
        
        # Step 2: Remove highly correlated features
        log("Step 2: Removing highly correlated features")
        correlated_features, corr_pairs = calculate_feature_correlation_matrix(
            self.df, numeric_features, correlation_threshold)
        numeric_features = [f for f in numeric_features if f not in correlated_features]
        after_collinearity = len(numeric_features)
        log(f"Removed {len(correlated_features)} highly correlated features, {after_collinearity} remaining")
        
        # Step 3: Remove unstable features (quick check)
        log("Step 3: Removing unstable features")
        unstable_features = quick_stability_check(
            self.df, numeric_features, self.target_col, n_splits=3, cv_threshold=cv_threshold)
        numeric_features = [f for f in numeric_features if f not in unstable_features]
        final_count = len(numeric_features)
        log(f"Removed {len(unstable_features)} unstable features, {final_count} remaining")
        
        # Store filtering results
        filtering_results = {
            'original_count': original_count,
            'after_constant_removal': after_constant,
            'after_collinearity_removal': after_collinearity,
            'final_count': final_count,
            'removed_constant': constant_features,
            'removed_collinear': correlated_features,
            'removed_unstable': unstable_features,
            'final_features': numeric_features
        }
        
        log(f"Feature filtering complete: {original_count} -> {final_count} features")
        end_section()
        return filtering_results

    def _remove_constant_features(self, numeric_features, threshold=1e-10):
        """Remove features with near-zero standard deviation"""
        constant_features = []
        for col in numeric_features:
            if self.df[col].std() <= threshold:
                constant_features.append(col)
        
        return constant_features

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

    def analyze_feature_statistics(self, selected_features=None):
        start_section("analyze_feature_statistics")
        log("Computing feature statistics")
        
        # Use enhanced selected features if available, otherwise use filtered features
        if selected_features is not None:
            numeric_features = selected_features
            log(f"Using {len(numeric_features)} enhanced selected features for statistics")
        else:
            numeric_features = self.feature_filtering_results['final_features']
            log(f"Using {len(numeric_features)} filtered features for statistics")
        
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
                
                # Save GPU model
                model_path = os.path.join(self.models_dir, f"gpu_random_forest_{self.target_col}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.saved_models[f"gpu_random_forest_{self.target_col}"] = model_path
                log(f"GPU Random Forest model saved to {model_path}")
                
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
        
        # Save CPU model using joblib for better sklearn compatibility
        model_path = os.path.join(self.models_dir, f"cpu_random_forest_{self.target_col}.joblib")
        joblib.dump(model, model_path)
        self.saved_models[f"cpu_random_forest_{self.target_col}"] = model_path
        log(f"CPU Random Forest model saved to {model_path}")
        
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

    def analyze_feature_importance(self, selected_features=None):
        start_section("analyze_feature_importance")
        
        # Use enhanced selected features if available, otherwise use filtered features
        if selected_features is not None:
            numeric_features = selected_features
            log(f"Analyzing importance for {len(numeric_features)} enhanced selected features")
        else:
            numeric_features = self.feature_filtering_results['final_features']
            log(f"Analyzing importance for {len(numeric_features)} filtered features")
        
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
        
    def analyze_shap_values(self, sample_size=100000, max_time_minutes=30, batch_size=500, save_partial=True, selected_features=None):
        """
        Calculate SHAP values for feature importance analysis with improved error handling
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
            # Use enhanced selected features if available, otherwise use filtered features
            if selected_features is not None:
                numeric_features = selected_features
                log(f"SHAP analysis for {len(numeric_features)} enhanced selected features")
            else:
                numeric_features = self.feature_filtering_results['final_features']
                log(f"SHAP analysis for {len(numeric_features)} filtered features")
            
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
            
            # Create CPU model for SHAP (SHAP works better with sklearn models)
            log("Training CPU model for SHAP analysis")
            if self.regression_mode:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
            log(f"Fitting model with {len(X_sample)} samples and {len(numeric_features)} features")
            fit_start = time.time()
            
            with tqdm.tqdm(total=1, desc="Training model") as pbar:
                model.fit(X_sample, y_sample)
                pbar.update(1)
                
            fit_time = time.time() - fit_start
            log(f"Model training completed in {fit_time:.2f} seconds")
            
            # Save the SHAP model too
            shap_model_path = os.path.join(self.models_dir, f"shap_model_{self.target_col}.joblib")
            joblib.dump(model, shap_model_path)
            self.saved_models[f"shap_model_{self.target_col}"] = shap_model_path
            log(f"SHAP model saved to {shap_model_path}")
            
            # Check memory again before SHAP calculation
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            log(f"Memory usage after model training: {current_memory:.1f}MB (growth: {memory_growth:.1f}MB)")
            
            # Create the SHAP explainer with timing
            log("Creating SHAP explainer")
            explainer_start = time.time()
            
            try:
                # Use TreeExplainer with safer parameters
                explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
                explainer_time = time.time() - explainer_start
                log(f"SHAP explainer created in {explainer_time:.2f} seconds")
            except Exception as e:
                log(f"Error creating SHAP explainer: {e}", LOG_LEVEL_ERROR)
                raise ValueError(f"Could not create SHAP explainer: {str(e)}")
            
            # Further reduce sample size for SHAP calculation if needed
            shap_sample_size = min(len(X_sample), 10000)  # Cap at 10,000 for SHAP calculation
            if len(X_sample) > shap_sample_size:
                log(f"Further sampling to {shap_sample_size} rows for SHAP calculation")
                shap_indices = np.random.choice(len(X_sample), shap_sample_size, replace=False)
                X_shap = X_sample.iloc[shap_indices].copy()
                # Free memory
                del X_sample
                gc.collect()
            else:
                X_shap = X_sample
            
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
            
            with tqdm.tqdm(total=total_rows, desc="SHAP calculation") as pbar:
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
                                    for idx, (feature, score) in enumerate(list(sorted_importance.items())[:30]):
                                        f.write(f"{idx+1}. {feature}: {score:.6f}\n")
                    
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

    def analyze_time_series_stability(self, n_splits=5, selected_features=None):
        start_section("analyze_time_series_stability")
        log("Analyzing time series stability with TimeSeriesSplit")
        
        # Use enhanced selected features if available, otherwise use filtered features
        if selected_features is not None:
            numeric_features = selected_features
            log(f"Using {len(numeric_features)} enhanced selected features for stability analysis")
        else:
            numeric_features = self.feature_filtering_results['final_features']
            log(f"Using {len(numeric_features)} filtered features for stability analysis")
        
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
            
            # Save model for this time split
            split_model_path = os.path.join(self.models_dir, f"time_split_{split_index}_{self.target_col}.joblib")
            joblib.dump(model, split_model_path)
            self.saved_models[f"time_split_{split_index}_{self.target_col}"] = split_model_path
            
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

def analyze_trading_dataset(file_path, target_col='long_signal', regression_mode=False, 
                         forecast_periods=14, run_shap=True, max_shap_time_minutes=30,
                         run_enhanced_selection=True, target_features=75):
    """
    Analyze trading dataset with enhanced feature selection and incremental file writing
    """
    start_section("analyze_trading_dataset")
    
    # Set up output file path using the PARAMETER target_col, not global
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
    selected_features = None
    
    try:
        # Dataset structure analysis
        log("Analyzing dataset structure")
        results['dataset_structure'] = analyzer.analyze_dataset_structure()
        write_partial_results(results, "Dataset Structure", file_path=output_file)
        
        # Feature filtering results
        log("Writing feature filtering results")
        results['feature_filtering'] = analyzer.feature_filtering_results
        write_partial_results(results, "Feature Filtering", file_path=output_file)
        
        # Enhanced feature selection
        if run_enhanced_selection:
            log("Running enhanced feature selection")
            results['enhanced_selection'] = analyzer.run_enhanced_feature_selection(
                target_features=target_features, validate=True)
            selected_features = results['enhanced_selection']['best_features']
            write_partial_results(results, "Enhanced Feature Selection", file_path=output_file)
            log(f"Enhanced selection complete: {len(selected_features)} features selected")
        else:
            log("Skipping enhanced feature selection")
            
    except Exception as e:
        log(f"Error during enhanced feature selection: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Run certain analyses in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit parallel tasks for non-dependent analyses
            stats_future = executor.submit(analyzer.analyze_feature_statistics, selected_features)
            patterns_future = executor.submit(analyzer.analyze_periodic_patterns)
            
            # Collect results as they complete
            log("Waiting for parallel analyses to complete...")
            
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
        results['feature_importance'] = analyzer.analyze_feature_importance(selected_features)
        write_partial_results(results, "Feature Importance", file_path=output_file)
    except Exception as e:
        log(f"Error during feature importance analysis: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Analyze SHAP values
        if run_shap:
            log("Starting SHAP analysis")
            results['shap_analysis'] = analyzer.analyze_shap_values(
                max_time_minutes=max_shap_time_minutes, selected_features=selected_features)
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
        results['time_series_stability'] = analyzer.analyze_time_series_stability(
            n_splits=5, selected_features=selected_features)
        write_partial_results(results, "Time Series Stability", file_path=output_file)
    except Exception as e:
        log(f"Error during time series stability analysis: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    try:
        # Write important features summary
        log("Writing important features")
        write_partial_results(results, "Important Features", file_path=output_file)
        
        # Write model information
        results['saved_models'] = analyzer.saved_models
        write_partial_results(results, "Model Information", file_path=output_file)
        
        # Also write to separate files
        with open(f'{target_col}_important_features.txt', 'w') as f:
            f.write("=== ENHANCED SELECTED FEATURES ===\n\n")
            f.write("# Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
            f.write(f"# Target: {target_col}\n")
            f.write(f"# Mode: {'Regression' if regression_mode else 'Classification'}\n")
            f.write(f"# Enhanced Selection: {'Enabled' if run_enhanced_selection else 'Disabled'}\n")
            
            if run_enhanced_selection and 'enhanced_selection' in results:
                f.write(f"# Selection Method: {results['enhanced_selection']['best_method']}\n")
                f.write(f"# Total Features: {len(results['enhanced_selection']['best_features'])}\n\n")
                
                for i, feature in enumerate(results['enhanced_selection']['best_features'], 1):
                    f.write(f"{i:2d}. {feature}\n")
            else:
                # Fallback to basic important features
                f.write("# Fallback to basic important features\n\n")
                important_features = set()
                for method_name in ['mutual_information', 'random_forest_importance']:
                    if method_name in results.get('feature_importance', {}):
                        method_dict = results['feature_importance'][method_name]
                        top_features = sorted(method_dict.items(), key=lambda x: x[1], reverse=True)[:target_features]
                        for feature, _ in top_features:
                            important_features.add(feature)
                
                if 'shap_importance' in results.get('shap_analysis', {}):
                    top_shap = sorted(results['shap_analysis']['shap_importance'].items(), 
                                     key=lambda x: x[1], reverse=True)[:target_features]
                    for feature, _ in top_shap:
                        important_features.add(feature)
                
                # Write to file
                for i, feature in enumerate(sorted(list(important_features)[:target_features]), 1):
                    f.write(f"{i:2d}. {feature}\n")
        
        # Write model summary
        with open(f'{target_col}_model_summary.txt', 'w') as f:
            f.write("=== SAVED MODELS SUMMARY ===\n\n")
            f.write("# Generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
            f.write(f"Target: {target_col}\n")
            f.write(f"Mode: {'Regression' if regression_mode else 'Classification'}\n")
            f.write(f"Enhanced Selection: {'Enabled' if run_enhanced_selection else 'Disabled'}\n")
            if run_enhanced_selection and 'enhanced_selection' in results:
                f.write(f"Selection Method: {results['enhanced_selection']['best_method']}\n")
                f.write(f"Selected Features: {len(results['enhanced_selection']['best_features'])}\n")
            f.write(f"Models directory: {analyzer.models_dir}\n\n")
            
            f.write("Saved models:\n")
            for model_name, model_path in analyzer.saved_models.items():
                f.write(f"  {model_name}: {model_path}\n")
            
            f.write(f"\nTo load a model:\n")
            f.write(f"import joblib\n")
            f.write(f"model = joblib.load('path_to_model.joblib')\n")
            f.write(f"# For pickle files: with open('path_to_model.pkl', 'rb') as f: model = pickle.load(f)\n")
                
    except Exception as e:
        log(f"Error writing important features: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
    
    # Write completion message
    with open(output_file, 'a') as f:
        f.write(f"\n{'='*30} ANALYSIS COMPLETED {'='*30}\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Enhanced Selection: {'Enabled' if run_enhanced_selection else 'Disabled'}\n")
        if run_enhanced_selection and 'enhanced_selection' in results:
            f.write(f"Best Selection Method: {results['enhanced_selection']['best_method']}\n")
            f.write(f"Selected Features: {len(results['enhanced_selection']['best_features'])}\n")
        f.write(f"Models saved in: {analyzer.models_dir}\n")
        f.write(f"Total models saved: {len(analyzer.saved_models)}\n")
    
    end_section()
    return results, df, analyzer

def run_configuration(config, file_path, run_shap, max_shap_time, run_enhanced_selection, target_features):
    """Function to run a single configuration"""
    target = config["target_col"]
    regression = config["regression_mode"]
    
    log(f"Starting analysis for target_col={target}, regression_mode={regression}")
    log(f"Enhanced selection: {'Enabled' if run_enhanced_selection else 'Disabled'}, Target features: {target_features}")
    
    try:
        results, df, analyzer = analyze_trading_dataset(
            file_path, 
            target_col=target,
            regression_mode=regression,
            run_shap=run_shap, 
            max_shap_time_minutes=max_shap_time,
            run_enhanced_selection=run_enhanced_selection,
            target_features=target_features
        )
        log(f"Analysis for {target} completed successfully")
        log(f"Models saved in directory: {analyzer.models_dir}")
        log(f"Number of models saved: {len(analyzer.saved_models)}")
        
        if run_enhanced_selection and 'enhanced_selection' in results:
            log(f"Enhanced selection method used: {results['enhanced_selection']['best_method']}")
            log(f"Features selected: {len(results['enhanced_selection']['best_features'])}")
        
        return True
    except Exception as e:
        log(f"Fatal error during analysis for {target}: {e}", LOG_LEVEL_ERROR)
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # File Path should be csv of all features

    file_path = r'./EURUSD_1min_sampled_indicators.csv'
    
    # Enhanced Feature Selection Configuration
    run_enhanced_selection = True  # Set to True to enable enhanced feature selection
    target_features = 75  # Number of features to select (as requested)
    
    # Set to False to skip SHAP if it's causing issues
    run_shap = True
    max_shap_time = 700  # minutes
    
    start_section("main")
    
    log(f"=== ENHANCED FEATURE SELECTION CONFIGURATION ===")
    log(f"Enhanced Selection: {'Enabled' if run_enhanced_selection else 'Disabled'}")
    log(f"Target Features: {target_features}")
    log(f"SHAP Analysis: {'Enabled' if run_shap else 'Disabled'}")
    log(f"Max SHAP Time: {max_shap_time} minutes")
    
    # Configuration to run
    configurations = [
        {"target_col": "long_signal", "regression_mode": False},
        {"target_col": "short_signal", "regression_mode": False},
        # {"target_col": "Close", "regression_mode": True}
    ]
    
    # Check GPU availability and force sequential execution if GPU is available
    cores_available = os.cpu_count()
    log(f"Detected {cores_available} CPU cores")
    log(f"GPU Available: {GPU_AVAILABLE}")
    
    # FORCE SEQUENTIAL EXECUTION WHEN GPU IS AVAILABLE
    if GPU_AVAILABLE:
        log("🔥 GPU detected - Running configurations SEQUENTIALLY to avoid CUDA context conflicts")
        run_parallel = False
    elif cores_available >= 6:
        log(f"No GPU detected - Running {len(configurations)} configurations in PARALLEL")
        run_parallel = True
    else:
        log("Limited cores - Running configurations SEQUENTIALLY")
        run_parallel = False
    
    if run_parallel:
        # Run in parallel with ProcessPoolExecutor (CPU only)
        with ProcessPoolExecutor(max_workers=len(configurations)) as executor:
            futures = []
            for config in configurations:
                futures.append(executor.submit(
                    run_configuration, config, file_path, run_shap, max_shap_time, 
                    run_enhanced_selection, target_features
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
        # Run sequentially (GPU-safe)
        log("Running configurations sequentially (GPU-safe mode)")
        for i, config in enumerate(configurations, 1):
            log(f"🚀 Starting configuration {i}/{len(configurations)}: {config['target_col']}")
            success = run_configuration(config, file_path, run_shap, max_shap_time, 
                                      run_enhanced_selection, target_features)
            status = "✅ SUCCESS" if success else "❌ FAILED"
            log(f"Configuration {config['target_col']} completed: {status}")
    
    print_timing_summary()
    end_section()
    
    log("All script executions completed")
    log(f"Execution mode: {'Sequential (GPU-safe)' if not run_parallel else 'Parallel (CPU-only)'}")
    log(f"Enhanced feature selection was: {'Enabled' if run_enhanced_selection else 'Disabled'}")
    log(f"Target features per analysis: {target_features}")
    
    # Final summary
    log("=== ANALYSIS SUMMARY ===")
    log("Files generated per target:")
    for config in configurations:
        target = config['target_col']
        log(f"  {target}:")
        log(f"    - {target}_analysis_results.txt (main results)")
        log(f"    - {target}_important_features.txt (selected features)")
        log(f"    - {target}_model_summary.txt (model information)")
        if run_enhanced_selection:
            log(f"    - {target}_enhanced_selection_details.txt (selection details)")
        if run_shap:
            log(f"    - {target}_partial_shap_results.txt (SHAP progress/results)")
        log(f"    - models_{target}_YYYYMMDD_HHMMSS/ (saved models directory)")
# %%