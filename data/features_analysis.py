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
        autocorr_values = {f'lag_{lag}': self.df['long_signal'].autocorr(lag) for lag in range(1, min(100, len(self.df) // 2))}
        top_correlations = sorted(autocorr_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        return {key: val for key, val in top_correlations}

    def analyze_feature_importance(self):
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.drop('Close', errors='ignore')
        
        X = self.df[numeric_features].copy()
        y = self.df['long_signal']
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
            return { 'random_forest_importance': {}} 
        

def analyze_trading_dataset(file_path):
    df = pd.read_csv(file_path)
    analyzer = TradingDataAnalyzer(df)

    results = {
        'dataset_structure': analyzer.analyze_dataset_structure(),
        'feature_statistics': analyzer.analyze_feature_statistics(),
        'periodic_patterns': analyzer.analyze_periodic_patterns(),
        'feature_importance': analyzer.analyze_feature_importance()
    }
    return results, df

def write_results_to_file(results, file_path='buy_class_analysis_results.txt'):
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

#%%
if __name__ == "__main__":
    # File Path should be csv of all features
    file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_features.csv'
    results, df = analyze_trading_dataset(file_path)
    write_results_to_file(results)

# %%
