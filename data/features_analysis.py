#%%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.utils.multiclass import type_of_target
import shap

target_col = 'short_signal'

class TradingDataAnalyzer:
    def __init__(self, df):
        self.df = df
        self.feature_stats = {}

    def analyze_dataset_structure(self):
        analysis = {
            'total_samples': len(self.df),
            'feature_count': len(self.df.columns),
            'memory_usage_MB': self.df.memory_usage().sum() / 1024**2,
            'missing_values': self.df.isnull().sum().sum(),
            'datatypes': self.df.dtypes.value_counts().to_dict()
        }

        buy_counts = self.df['long_signal'].value_counts()
        short_counts = self.df['short_signal'].value_counts()
        total = len(self.df)
        buy_ratio = buy_counts.get(1, 0) / total
        sell_ratio = short_counts.get(1, 0) / total

        analysis['signal_distribution'] = {
            'buy_signals': int(buy_counts.get(1, 0)),
            'short_signals': int(short_counts.get(1, 0)),
            'no_buy_signals': int(buy_counts.get(0, 0)),
            'no_short_signals': int(short_counts.get(0, 0)),
            'buy_signal_ratio': buy_ratio,
            'short_signal_ratio': sell_ratio,
            'sell_imbalance_warning': sell_ratio < 0.1 or sell_ratio > 0.9,
            'buy_imbalance_warning': buy_ratio < 0.1 or buy_ratio > 0.9
        }
        return analysis

    def analyze_feature_statistics(self):
        numeric_features = self.df.select_dtypes(include=[np.number]).columns
        for feature in numeric_features:
            stats_data = {
                'mean': self.df[feature].mean(),
                'median': self.df[feature].median(),
                'std': self.df[feature].std(),
                'skew': stats.skew(self.df[feature].dropna()),
                'kurtosis': stats.kurtosis(self.df[feature].dropna()),
                'unique_values': self.df[feature].nunique(),
                'iqr': self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25),
                'outlier_percentage': np.mean((self.df[feature] < self.df[feature].quantile(0.25) - 1.5 * (self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25))) |
                                          (self.df[feature] > self.df[feature].quantile(0.75) + 1.5 * (self.df[feature].quantile(0.75) - self.df[feature].quantile(0.25)))) * 100
            }
            self.feature_stats[feature] = stats_data
        return self.feature_stats

    def analyze_periodic_patterns(self):
        autocorr_values = {f'lag_{lag}': self.df[target_col].autocorr(lag) for lag in range(1, min(100, len(self.df) // 2))}
        top_correlations = sorted(autocorr_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        return {key: val for key, val in top_correlations}

    def analyze_feature_importance(self):
        exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close'}
        numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
        
        X = self.df[numeric_features].copy()
        y = self.df[target_col]
        target_type = type_of_target(y)
        # Handle Inf and NaN values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        # Clip extreme values
        X = X.clip(lower=np.finfo(np.float64).min / 2, upper=np.finfo(np.float64).max / 2)

        # Scale features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

        # Ensure y is correctly formatted
        y = y.values.ravel()

        # Feature importance calculation
        try:
            if target_type in ["continuous", "continuous-multioutput"]:
                # Regression
                mi_scores = mutual_info_regression(X, y, discrete_features=False, random_state=42)
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("Using regression model")
            else:
                # Classification
                mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("Using classification model")

            model.fit(X, y)

            mi_results = {feature: score for feature, score in zip(numeric_features, mi_scores)}
            rf_importance = {feature: importance for feature, importance in zip(numeric_features, model.feature_importances_)}

            return {
                'mutual_information': dict(sorted(mi_results.items(), key=lambda x: x[1], reverse=True)),
                'random_forest_importance': dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True))
            }
        except Exception as e:
            print(f"Error during feature importance calculation: {e}")
            return {'random_forest_importance': {}} 

    def analyze_shap_values(self, sample_size=1000):
        """
        Calculate SHAP values for feature importance analysis.
        
        Args:
            sample_size: Number of samples to use for SHAP analysis (to manage computational load)
        
        Returns:
            Dictionary with SHAP values and summary statistics
        """
        try:
            exclude_columns = {'short_signal', 'long_signal', 'close_position', 'Close'}
            numeric_features = [col for col in self.df.select_dtypes(include=[np.number]).columns if col not in exclude_columns]
            
            # Prepare data
            X = self.df[numeric_features].copy()
            y = self.df[target_col]
            target_type = type_of_target(y)
            
            # Handle problematic values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            X.fillna(X.median(), inplace=True)
            X = X.clip(lower=np.finfo(np.float64).min / 2, upper=np.finfo(np.float64).max / 2)
            
            # If dataset is large, take a sample to reduce computation time
            if len(X) > sample_size:
                sample_indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[sample_indices]
                y_sample = y.iloc[sample_indices]
            else:
                X_sample = X
                y_sample = y
            
            # Create the appropriate model
            if target_type in ["continuous", "continuous-multioutput"]:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                print("Using regression model for SHAP analysis")
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                print("Using classification model for SHAP analysis")
            
            # Fit the model
            model.fit(X_sample, y_sample)
            
            # Create the SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            shap_values = explainer.shap_values(X_sample)

            # Check what type of output SHAP gave
            if isinstance(shap_values, list):
                # List of (n_samples, n_features) per class
                if len(shap_values) == 2:
                    # Binary classification — use class 1
                    shap_values = shap_values[1]
                else:
                    # Multi-class: average across classes
                    shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Newer SHAP: shape (n_samples, n_features, n_classes)
                shap_values = np.mean(np.abs(shap_values), axis=2)
            else:
                # Regression: (n_samples, n_features)
                shap_values = np.abs(shap_values)
            # Mean SHAP value per feature (1D array of shape: [n_features])
            mean_abs_shap = shap_values.mean(axis=0)

            # Normalize to sum to 1 (optional, makes it comparable to RF)
            shap_total = np.sum(mean_abs_shap)
            mean_abs_shap_normalized = mean_abs_shap / shap_total

            # Create feature:importance dictionary
            shap_importance = {
                feature: float(importance)
                for feature, importance in zip(numeric_features, mean_abs_shap_normalized)
            }

            # # If mean_abs_shap is 2D, reduce it to 1D
            # if mean_abs_shap.ndim > 1:
            #     mean_abs_shap = mean_abs_shap.mean(axis=0)

            # # Ensure it's 1D before zipping with features
            # shap_importance = {feature: float(importance) for feature, importance in zip(numeric_features, mean_abs_shap)}
            # # print(f"shap_values shape: {np.shape(shap_normalized)}")
            # print(f"mean_abs_shap shape: {np.shape(mean_abs_shap)}")

            # Return sorted results
            return {
                'shap_importance': dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            }
            
        except Exception as e:
            print(f"Error during SHAP analysis: {e}")
            import traceback
            traceback.print_exc()
            return {'shap_importance': {}}

def analyze_trading_dataset(file_path):
    df = pd.read_csv(file_path)
    analyzer = TradingDataAnalyzer(df)

    results = {
        'dataset_structure': analyzer.analyze_dataset_structure(),
        'feature_statistics': analyzer.analyze_feature_statistics(),
        'periodic_patterns': analyzer.analyze_periodic_patterns(),
        'feature_importance': analyzer.analyze_feature_importance(),
        'shap_analysis': analyzer.analyze_shap_values()
    }
    return results, df

def write_results_to_file(results, file_path=f'{target_col}trading_analysis_results.txt'):
    important_features = set()
    
    # Consider features important if they score high in any method
    feature_importance_methods = [
        ('mutual_information', 0.0027),
        ('random_forest_importance', 0.0027),
        ('shap_importance', 0.0027)
    ]
    
    for method_name, threshold in feature_importance_methods:
        if method_name == 'shap_importance':
            method_dict = results['shap_analysis']
        else:
            method_dict = results['feature_importance']
        
        for feature, score in method_dict.get(method_name, {}).items():
            if score > threshold:
                important_features.add(feature)

    with open(f'{target_col}_important_features.txt', 'w') as f:
        for feature in sorted(important_features):
            f.write(feature + '\n')

    with open(file_path, 'w') as f:
        f.write("=== TRADING DATASET ANALYSIS ===\n\n")
        
        f.write("Dataset Structure:\n" + "-" * 50 + "\n")
        for key, value in results['dataset_structure'].items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nFeature Statistics:\n" + "-" * 50 + "\n")
        for feature, stats in results['feature_statistics'].items():
            f.write(f"{feature}: {stats}\n")

        f.write("\nTop Periodic Patterns:\n" + "-" * 50 + "\n")
        for lag, correlation in results['periodic_patterns'].items():
            f.write(f"{lag}: {correlation:.4f}\n")
        
        f.write("\nFeature Importance:\n" + "-" * 50 + "\n")
        f.write("\nMutual Information Scores:\n")
        for feature, score in results['feature_importance']['mutual_information'].items():
            f.write(f"{feature}: {score:.4f}\n")

        f.write("\nRandom Forest Feature Importance:\n")
        for feature, score in results['feature_importance']['random_forest_importance'].items():
            f.write(f"{feature}: {score:.4f}\n")
            
        f.write("\nSHAP Feature Importance:\n")
        for feature, score in results['shap_analysis']['shap_importance'].items():
            f.write(f"{feature}: {score:.4f}\n")

        # Add a section comparing top features across methods
        f.write("\nTop 10 Features by Importance Method:\n" + "-" * 50 + "\n")
        
        importance_methods = {
            'Mutual Information': results['feature_importance'].get('mutual_information', {}),
            'Random Forest': results['feature_importance'].get('random_forest_importance', {}),
            'SHAP Values': results['shap_analysis'].get('shap_importance', {})
        }
        
        # Get top 10 from each method
        top_features = {}
        for method_name, importance_dict in importance_methods.items():
            top_features[method_name] = list(importance_dict.keys())[:10]
        
        # Print comparison table
        f.write(f"{'Rank':<5}{'Mutual Information':<25}{'Random Forest':<25}{'SHAP Values':<25}\n")
        f.write("-" * 80 + "\n")
        
        for i in range(min(10, max(len(features) for features in top_features.values()))):
            row = f"{i+1:<5}"
            for method in ['Mutual Information', 'Random Forest', 'SHAP Values']:
                feature = top_features[method][i] if i < len(top_features[method]) else ""
                score = importance_methods[method].get(feature, 0)
                if feature:
                    row += f"{feature} ({score:.4f})"[:24].ljust(25)
                else:
                    row += " " * 25
            f.write(row + "\n")

#%%
if __name__ == "__main__":
    # File Path should be csv of all features
    file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_features.csv'
    results, df = analyze_trading_dataset(file_path)
    write_results_to_file(results)

# %%