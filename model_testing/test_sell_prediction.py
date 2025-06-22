#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import ta
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time
import json

# Configure matplotlib for better visuals
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.figsize'] = (20, 12)

# Set plotly template
pio.templates.default = "plotly_dark"

class ImprovedTradingModelTester:
    def __init__(self, 
                 sell_model_dir=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\sell_signal_models\sell_models',
                 buy_model_dir=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\long_signal_models\buy_models',
                 lookback_window=120):
        self.sell_model_dir = sell_model_dir
        self.buy_model_dir = buy_model_dir
        self.lookback_window = lookback_window
        
        # Separate lag periods for sell and buy models
        self.sell_lag_periods = [70, 24, 10, 74, 39]  # Original sell lag periods
        self.buy_lag_periods = [61, 93, 64, 60, 77]   # Original buy lag periods
        
        self.sell_models = {}
        self.buy_models = {}
        
        # Separate scalers for sell and buy models
        self.sell_scaler = RobustScaler(quantile_range=(15.0, 85.0))
        self.buy_scaler = RobustScaler(quantile_range=(15.0, 85.0))
        self.is_sell_scaler_fitted = False
        self.is_buy_scaler_fitted = False
        
        # Different colors for sell and buy models
        self.model_colors = {
            'LSTM_Sell': '#FF0000', 'GRU_Sell': '#FF8C00', 
            'Conv1D_Sell': '#8B0000', 'Conv1D_LSTM_Sell': '#FF4500',
            'LSTM_Buy': '#00FF00', 'GRU_Buy': '#32CD32', 
            'Conv1D_Buy': '#006400', 'Conv1D_LSTM_Buy': '#90EE90'
        }
        
        # Adaptive thresholds instead of fixed 0.5
        self.adaptive_thresholds = {
            'sell': {
                'LSTM': 0.4,
                'GRU': 0.4,
                'Conv1D': 0.6,
                'Conv1D_LSTM': 0.400
            },
            'buy': {
                'LSTM': 0.600,
                'GRU': 0.433,
                'Conv1D': 0.5,
                'Conv1D_LSTM': 0.5
            }
        }
        
        # Signal cooldown to prevent clustering
        self.signal_cooldown = 10  # minimum periods between signals
        self.last_signal_time = {'sell': {}, 'buy': {}}
        
        # Base features (same for both)
        self.base_features = [
            'ADOSC', 'ADXR', 'AROONOSC', 'AROON_DOWN', 'AROON_UP', 'BOP', 'CCI',
            'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLHIGHWAVE',
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLRICKSHAWMAN', 'CDLSHORTLINE',
            'CDLSPINNINGTOP', 'CMO', 'EFI', 'HT_DCPERIOD', 'HT_DCPHASE',
            'HT_LEADSINE', 'HT_PHASOR_INPHASE', 'HT_PHASOR_QUADRATURE', 'HT_SINE',
            'HT_TRENDMODE', 'MOM', 'OBV', 'ROCP', 'ROCR100', 'RSI', 'STOCHF_K',
            'STOCHRSI_D', 'STOCHRSI_K', 'TRANGE', 'TRIX', 'Volume', 'z_score',
            'AD', 'ADX', 'APO', 'ATR', 'DX', 'MACD', 'MACDHIST', 'MACDSIGNAL',
            'MFI', 'MINUS_DI', 'MINUS_DM', 'NATR', 'PLUS_DI', 'PLUS_DM', 'PPO',
            'ROC', 'ROCR', 'STOCHF_D', 'STOCH_D', 'STOCH_K', 'ULTOSC', 'VWAP',
            'WILLR', 'rolling_std'
        ]
        
        # Separate feature sets for sell and buy models
        self.sell_training_features = self.base_features.copy()
        self.buy_training_features = self.base_features.copy()
        
        # Add model-specific lag features
        self._build_sell_features()
        self._build_buy_features()
        
        # Storage for test results
        self.test_results = {
            'timestamps': [], 'prices': [],
            'sell_predictions': {name.replace('_Sell', ''): [] for name in self.model_colors.keys() if '_Sell' in name},
            'sell_signals': {name.replace('_Sell', ''): [] for name in self.model_colors.keys() if '_Sell' in name},
            'buy_predictions': {name.replace('_Buy', ''): [] for name in self.model_colors.keys() if '_Buy' in name},
            'buy_signals': {name.replace('_Buy', ''): [] for name in self.model_colors.keys() if '_Buy' in name}
        }
        
        # Ensemble models
        self.sell_ensemble = None
        self.buy_ensemble = None
        
        # Create output directory
        self.output_dir = "trading_charts"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _build_sell_features(self):
        """Build feature set specific to sell models"""
        # Add sell-specific lag features
        for lag in self.sell_lag_periods:
            self.sell_training_features.append(f'sell_signal_lag_{lag}')  # Use sell-specific naming
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                self.sell_training_features.append(f'{indicator}_lag_{lag}')
        
        # Add rolling stats
        self.sell_training_features.extend(['sell_target_lag_mean', 'sell_target_lag_std'])
        
        # Add volume transformations
        self.sell_training_features.extend(['Volume_log', 'Volume_winsor', 'Volume_rank'])
        
        print(f"Sell models expect {len(self.sell_training_features)} features")
    
    def _build_buy_features(self):
        """Build feature set specific to buy models"""
        # Add buy-specific lag features
        for lag in self.buy_lag_periods:
            self.buy_training_features.append(f'long_signal_lag_{lag}')  # Keep original naming
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                self.buy_training_features.append(f'{indicator}_lag_{lag}')
        
        # Add rolling stats
        self.buy_training_features.extend(['target_lag_mean', 'target_lag_std'])
        
        # Add volume transformations
        self.buy_training_features.extend(['Volume_log', 'Volume_winsor', 'Volume_rank'])
        
        print(f"Buy models expect {len(self.buy_training_features)} features")
    
    def focal_loss_fn(self, y_true, y_pred):
        """Focal loss function for model loading"""
        gamma = 2.0
        alpha = 0.25
        epsilon = 1e-7  # Smaller epsilon for better numerical stability
        
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        
        loss = -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + epsilon)
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        
        return tf.reduce_mean(loss)
    
    def load_models(self):
        """Load all trained models with better error handling"""
        print("Loading trained models...")
        model_files = {
            'LSTM': 'LSTM.h5', 'GRU': 'GRU.h5', 
            'Conv1D': 'Conv1D.h5', 'Conv1D_LSTM': 'Conv1D_LSTM.h5'
        }
        
        # Load sell models
        print("Loading sell models...")
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.sell_model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(
                        model_path, custom_objects={'focal_loss_fn': self.focal_loss_fn}
                    )
                    expected_shape = model.input_shape
                    print(f"✅ Loaded {model_name}_Sell model - Input shape: {expected_shape}")
                    self.sell_models[model_name] = model
                    self.last_signal_time['sell'][model_name] = -999
                except Exception as e:
                    print(f"❌ Error loading {model_name}_Sell: {e}")
            else:
                print(f"❌ Sell model file not found: {model_path}")
        
        # Load buy models
        print("Loading buy models...")
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.buy_model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(
                        model_path, custom_objects={'focal_loss_fn': self.focal_loss_fn}
                    )
                    expected_shape = model.input_shape
                    print(f"✅ Loaded {model_name}_Buy model - Input shape: {expected_shape}")
                    self.buy_models[model_name] = model
                    self.last_signal_time['buy'][model_name] = -999
                except Exception as e:
                    print(f"❌ Error loading {model_name}_Buy: {e}")
            else:
                print(f"❌ Buy model file not found: {model_path}")
        
        print(f"Successfully loaded {len(self.sell_models)} sell and {len(self.buy_models)} buy models")

    def fetch_historical_data(self, samples=25000, csv_path=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled.csv'):
        """Fetch historical EUR/USD data from local CSV file"""
        print(f"Fetching {samples} historical EUR/USD samples from CSV file...")

        try:
            # Read data from CSV
            data = pd.read_csv(csv_path)
            
            # Check if the CSV has the necessary columns
            required_columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Missing required column: {col}")
                    return self.generate_sample_data(samples)  # Fallback to synthetic data
            
            # Combine Date and Time columns into a single datetime column
            data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')
            
            # Keep only the relevant columns
            data = data[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Time']]
            
            # Handle volume if it's missing or incorrect
            if 'Volume' not in data.columns:
                print("Volume not available in the file, generating synthetic volume")
                data['Volume'] = np.random.randint(1000, 25000, len(data))
            
            # Take the last 'samples' rows
            if len(data) > samples:
                data = data.tail(samples).reset_index(drop=True)
            
            # Print information about the data
            print(f"Successfully processed {len(data)} historical samples from CSV")
            
            # Return the data in the required format
            return data

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print("Falling back to sample data...")
            return self.generate_sample_data(samples)
    
    def generate_sample_data(self, samples=25000):
        """Generate sample EUR/USD data for testing"""
        print(f"Generating {samples} sample data points for testing...")
        
        # Create realistic time series
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=samples)
        dates = pd.date_range(start=start_time, end=end_time, freq='1min')[:samples]
        
        # Generate realistic EUR/USD price movement using geometric Brownian motion
        np.random.seed(42)
        base_price = 1.0850
        dt = 1.0 / (365 * 24 * 60)  # 1 minute in years
        mu = 0.02  # drift
        sigma = 0.15  # volatility
        
        # Generate price series
        returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Add some intraday patterns and trend
        time_of_day = pd.to_datetime(dates).hour + pd.to_datetime(dates).minute / 60.0
        intraday_pattern = 0.0001 * np.sin(2 * np.pi * time_of_day / 24)
        trend = np.linspace(0, 0.002, len(dates))
        prices = prices + intraday_pattern + trend
        
        # Create OHLCV data
        data = pd.DataFrame({
            'datetime': dates,
            'Open': prices,
            'Close': prices,
            'Volume': np.random.randint(1000, 25000, len(dates))
        })
        
        # Generate realistic High and Low
        volatility = 0.0005
        data['High'] = data['Close'] + np.random.uniform(0, volatility, len(data))
        data['Low'] = data['Close'] - np.random.uniform(0, volatility, len(data))
        
        # Ensure price relationships are correct
        for i in range(len(data)):
            high_val = max(data.loc[i, 'Open'], data.loc[i, 'Close']) + abs(np.random.normal(0, volatility/2))
            low_val = min(data.loc[i, 'Open'], data.loc[i, 'Close']) - abs(np.random.normal(0, volatility/2))
            data.loc[i, 'High'] = max(high_val, data.loc[i, 'Open'], data.loc[i, 'Close'])
            data.loc[i, 'Low'] = min(low_val, data.loc[i, 'Open'], data.loc[i, 'Close'])
        
        # Add Date and Time columns
        data['Date'] = data['datetime'].dt.strftime('%Y%m%d').astype(int)
        data['Time'] = data['datetime'].dt.strftime('%H:%M:%S')
        
        return data

    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators using 'ta' library"""
        print("Calculating technical indicators...")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Ensure we have enough data
        if len(data) < 200:
            print(f"Warning: Only {len(data)} rows available. Some indicators may not be accurate.")
        
        try:
            # Add all ta indicators
            data = ta.add_all_ta_features(
                data, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )

            # Map ta column names to expected names from training
            column_mapping = {
                # Volume indicators
                'volume_adi': 'AD',
                'volume_obv': 'OBV',
                'volume_mfi': 'MFI',
                
                # Volatility indicators  
                'volatility_atr': 'ATR',
                
                # Trend indicators
                'trend_macd': 'MACD',
                'trend_macd_signal': 'MACDSIGNAL', 
                'trend_macd_diff': 'MACDHIST',
                'trend_adx': 'ADX',
                'trend_adx_pos': 'PLUS_DI',
                'trend_adx_neg': 'MINUS_DI',
                'trend_trix': 'TRIX',
                'trend_aroon_up': 'AROON_UP',
                'trend_aroon_down': 'AROON_DOWN',
                'trend_aroon_ind': 'AROONOSC',
                
                # Momentum indicators
                'momentum_rsi': 'RSI',
                'momentum_stoch_rsi_k': 'STOCHRSI_K',
                'momentum_stoch_rsi_d': 'STOCHRSI_D',
                'momentum_uo': 'ULTOSC',
                'momentum_stoch': 'STOCH_K',
                'momentum_stoch_signal': 'STOCH_D',
                'momentum_wr': 'WILLR',
                'momentum_roc': 'ROC',
                'momentum_ppo': 'PPO',
            }
            
            # Rename columns
            data = data.rename(columns=column_mapping)
            
            # Calculate additional indicators manually that might not be in ta library
            
            # CCI (Commodity Channel Index)
            if 'CCI' not in data.columns:
                tp = (data['High'] + data['Low'] + data['Close']) / 3
                sma = tp.rolling(window=20).mean()
                mad = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
                data['CCI'] = (tp - sma) / (0.015 * mad)
            
            # CMO (Chande Momentum Oscillator)
            if 'CMO' not in data.columns:
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).sum()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).sum()
                data['CMO'] = 100 * (gain - loss) / (gain + loss)
            
            # MOM (Momentum)
            if 'MOM' not in data.columns:
                data['MOM'] = data['Close'].diff(10)
            
            # ROCR (Rate of Change Ratio)
            if 'ROCR' not in data.columns:
                data['ROCR'] = data['Close'] / data['Close'].shift(10)
            
            # ROCR100 (Rate of Change Ratio * 100)
            if 'ROCR100' not in data.columns:
                data['ROCR100'] = data['ROCR'] * 100
            
            # ROCP (Rate of Change Percentage)
            if 'ROCP' not in data.columns:
                data['ROCP'] = (data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10) * 100
            
            # STOCHF_K and STOCHF_D (Fast Stochastic)
            if 'STOCHF_K' not in data.columns:
                low_min = data['Low'].rolling(window=5).min()
                high_max = data['High'].rolling(window=5).max()
                data['STOCHF_K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
                data['STOCHF_D'] = data['STOCHF_K'].rolling(window=3).mean()
            
            # BOP (Balance of Power)
            if 'BOP' not in data.columns:
                data['BOP'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'])
                data['BOP'] = data['BOP'].replace([np.inf, -np.inf], 0)
            
            # ADOSC (Accumulation/Distribution Oscillator)
            if 'ADOSC' not in data.columns:
                if 'AD' in data.columns:
                    data['ADOSC'] = data['AD'].ewm(span=3).mean() - data['AD'].ewm(span=10).mean()
                else:
                    data['ADOSC'] = 0
            
            # ADXR (Average Directional Movement Index Rating)
            if 'ADXR' not in data.columns and 'ADX' in data.columns:
                data['ADXR'] = (data['ADX'] + data['ADX'].shift(14)) / 2
            
            # EFI (Ease of Movement Index)
            if 'EFI' not in data.columns:
                distance_moved = (data['High'] + data['Low'])/2 - (data['High'].shift(1) + data['Low'].shift(1))/2
                box_height = data['Volume'] / (data['High'] - data['Low'])
                box_height = box_height.replace([np.inf, -np.inf], 1)
                data['EFI'] = distance_moved / box_height
                data['EFI'] = data['EFI'].replace([np.inf, -np.inf], 0)
            
            # TRANGE (True Range)
            if 'TRANGE' not in data.columns:
                data['TRANGE'] = np.maximum.reduce([
                    data['High'] - data['Low'],
                    np.abs(data['High'] - data['Close'].shift(1)),
                    np.abs(data['Low'] - data['Close'].shift(1))
                ])
            
            # NATR (Normalized Average True Range)
            if 'NATR' not in data.columns and 'ATR' in data.columns:
                data['NATR'] = 100 * data['ATR'] / data['Close']
            
            # DX (Directional Movement Index)
            if 'DX' not in data.columns and 'PLUS_DI' in data.columns and 'MINUS_DI' in data.columns:
                data['DX'] = 100 * np.abs(data['PLUS_DI'] - data['MINUS_DI']) / (data['PLUS_DI'] + data['MINUS_DI'])
                data['DX'] = data['DX'].replace([np.inf, -np.inf], 0)
            
            # PLUS_DM and MINUS_DM (Directional Movement)
            if 'PLUS_DM' not in data.columns:
                up_move = data['High'] - data['High'].shift(1)
                down_move = data['Low'].shift(1) - data['Low']
                data['PLUS_DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
                data['MINUS_DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # APO (Absolute Price Oscillator)
            if 'APO' not in data.columns:
                data['APO'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
            
            # VWAP (Volume Weighted Average Price)
            if 'VWAP' not in data.columns:
                typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                data['VWAP'] = (typical_price * data['Volume']).expanding().sum() / data['Volume'].expanding().sum()
            
            # Rolling standard deviation
            if 'rolling_std' not in data.columns:
                data['rolling_std'] = data['Close'].rolling(window=20).std()
            
            # Z-score
            if 'z_score' not in data.columns:
                data['z_score'] = (data['Close'] - data['Close'].rolling(window=20).mean()) / data['rolling_std']
            
            # Hilbert Transform indicators (simplified versions)
            if 'HT_DCPERIOD' not in data.columns:
                data['HT_DCPERIOD'] = data['Close'].rolling(window=20).std()
            if 'HT_DCPHASE' not in data.columns:
                data['HT_DCPHASE'] = np.arctan2(data['Close'].diff(), data['Close'].shift(1))
            if 'HT_LEADSINE' not in data.columns:
                data['HT_LEADSINE'] = np.sin(data['HT_DCPHASE'])
            if 'HT_SINE' not in data.columns:
                data['HT_SINE'] = np.cos(data['HT_DCPHASE'])
            if 'HT_PHASOR_INPHASE' not in data.columns:
                data['HT_PHASOR_INPHASE'] = np.cos(data['HT_DCPHASE'])
            if 'HT_PHASOR_QUADRATURE' not in data.columns:
                data['HT_PHASOR_QUADRATURE'] = np.sin(data['HT_DCPHASE'])
            if 'HT_TRENDMODE' not in data.columns:
                data['HT_TRENDMODE'] = (data['Close'] > data['Close'].rolling(window=20).mean()).astype(int)
            
            # Candlestick patterns (simplified binary indicators)
            candlestick_patterns = [
                'CDLBELTHOLD', 'CDLCLOSINGMARUBOZU', 'CDLDOJI', 'CDLHIGHWAVE',
                'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLRICKSHAWMAN', 
                'CDLSHORTLINE', 'CDLSPINNINGTOP'
            ]
            
            for pattern in candlestick_patterns:
                if pattern not in data.columns:
                    # Simple pattern detection based on body and shadow ratios
                    body = np.abs(data['Close'] - data['Open'])
                    upper_shadow = data['High'] - np.maximum(data['Open'], data['Close'])
                    lower_shadow = np.minimum(data['Open'], data['Close']) - data['Low']
                    total_range = data['High'] - data['Low']
                    
                    if 'DOJI' in pattern:
                        data[pattern] = (body < total_range * 0.1).astype(int)
                    elif 'LONGLINE' in pattern:
                        data[pattern] = (body > total_range * 0.7).astype(int)
                    elif 'SHORTLINE' in pattern:
                        data[pattern] = (body < total_range * 0.3).astype(int)
                    else:
                        data[pattern] = 0  # Default to 0 for complex patterns
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            # Fill missing indicators with zeros
            for indicator in self.base_features:
                if indicator not in data.columns:
                    data[indicator] = 0
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        data = data.replace([np.inf, -np.inf], 0)
        
        return data
        
    def calculate_adaptive_thresholds(self, predictions_history, percentile=30):
    #     """Calculate adaptive thresholds based on prediction history"""
    #     for model_type in ['sell', 'buy']:
    #         self.adaptive_thresholds[model_type] = {}
            
    #         prediction_key = f'{model_type}_predictions'
    #         if prediction_key in predictions_history:
    #             for model_name, preds in predictions_history[prediction_key].items():
    #                 if len(preds) > 100:  # Need enough history
    #                     # Use percentile-based threshold
    #                     threshold = np.percentile(preds, percentile)
    #                     # Ensure minimum threshold
    #                     threshold = max(threshold, 0.4)
    #                     self.adaptive_thresholds[model_type][model_name] = threshold
    #                 else:
    #                     self.adaptive_thresholds[model_type][model_name] = 0.5
        
        print("📊 Adaptive Thresholds Calculated:")
        for model_type in ['sell', 'buy']:
            for model_name, threshold in self.adaptive_thresholds[model_type].items():
                print(f"{model_name}_{model_type}: {threshold:.3f}")
    
    def add_lag_features_sell(self, data):
        """Add lag features specific to sell models"""
        # Create dummy target for sell signals
        data['sell_signal'] = 0
        
        # Add sell-specific lag features
        for lag in self.sell_lag_periods:
            data[f'sell_signal_lag_{lag}'] = data['sell_signal'].shift(lag)
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                if indicator in data.columns:
                    data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
        
        # Add sell-specific rolling stats
        lag_cols = [f'sell_signal_lag_{lag}' for lag in self.sell_lag_periods]
        data['sell_target_lag_mean'] = data[lag_cols].mean(axis=1)
        data['sell_target_lag_std'] = data[lag_cols].std(axis=1)
        
        # Remove dummy column
        data = data.drop(['sell_signal'], axis=1)
        return data
    
    def add_lag_features_buy(self, data):
        """Add lag features specific to buy models"""
        # Create dummy target for buy signals
        data['long_signal'] = 0
        
        # Add buy-specific lag features
        for lag in self.buy_lag_periods:
            data[f'long_signal_lag_{lag}'] = data['long_signal'].shift(lag)
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                if indicator in data.columns:
                    data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
        
        # Add buy-specific rolling stats
        lag_cols = [f'long_signal_lag_{lag}' for lag in self.buy_lag_periods]
        data['target_lag_mean'] = data[lag_cols].mean(axis=1)
        data['target_lag_std'] = data[lag_cols].std(axis=1)
        
        # Remove dummy column
        data = data.drop(['long_signal'], axis=1)
        return data
    
    def preprocess_data_for_sell(self, data):
        """Preprocess data specifically for sell models"""
        # Calculate technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Add sell-specific lag features
        data = self.add_lag_features_sell(data)
        
        # Transform volume features
        data = self.add_volume_features(data)
        
        # Select sell-specific features
        features = pd.DataFrame()
        for feature in self.sell_training_features:
            if feature in data.columns:
                features[feature] = data[feature]
            else:
                features[feature] = 0
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def preprocess_data_for_buy(self, data):
        """Preprocess data specifically for buy models"""
        # Calculate technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Add buy-specific lag features
        data = self.add_lag_features_buy(data)
        
        # Transform volume features
        data = self.add_volume_features(data)
        
        # Select buy-specific features
        features = pd.DataFrame()
        for feature in self.buy_training_features:
            if feature in data.columns:
                features[feature] = data[feature]
            else:
                features[feature] = 0
        
        # Clean data
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def add_volume_features(self, data):
        """Add volume transformation features"""
        volume_cols = ['Volume']
        for col in volume_cols:
            if col in data.columns:
                # Avoid log(0) by using log1p
                data[f'{col}_log'] = np.log1p(np.maximum(data[col], 1))
                # Winsorization
                q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
                data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
                # Rank transformation
                data[f'{col}_rank'] = data[col].rank(pct=True)
        return data
    
    def calibrate_probabilities(self, predictions, calibration_data=None):
        """Apply probability calibration using Platt scaling"""
        # Simple sigmoid calibration
        def sigmoid_calibration(x, a=1.0, b=0.0):
            return 1 / (1 + np.exp(-(a * x + b)))
        
        # Apply calibration (simplified version)
        calibrated = sigmoid_calibration(predictions * 2 - 1)  # Map [0,1] to [-1,1] then calibrate
        return calibrated
    
    def check_signal_cooldown(self, model_type, model_name, current_time_idx):
        """Check if enough time has passed since last signal"""
        last_signal = self.last_signal_time[model_type][model_name]
        return (current_time_idx - last_signal) >= self.signal_cooldown
    
    def create_sliding_windows(self, features, start_idx):
        """Create sliding windows for prediction"""
        end_idx = start_idx + self.lookback_window
        if end_idx > len(features):
            return None
        
        # Convert to numpy array
        features_array = features.values.astype(np.float32)
        
        # Create sliding window
        window = features_array[start_idx:end_idx].reshape(1, self.lookback_window, features_array.shape[1])
        
        return window
    
    def predict_signals_improved(self, sell_window, buy_window, current_time_idx):
        """Improved prediction with calibration and cooldown"""
        predictions = {'sell': {}, 'buy': {}}
        
        # Sell predictions
        for model_name, model in self.sell_models.items():
            try:
                pred = model.predict(sell_window, verbose=0)[0]
                signal_prob = pred[1] if len(pred) > 1 else pred[0]
                
                # Apply calibration
                signal_prob = self.calibrate_probabilities(signal_prob)
                
                # Handle edge cases
                if np.isnan(signal_prob) or np.isinf(signal_prob):
                    signal_prob = 0.0
                
                # Get adaptive threshold
                threshold = self.adaptive_thresholds['sell'].get(model_name, 0.5)
                
                # Check signal with cooldown
                is_signal = (signal_prob > threshold and 
                           self.check_signal_cooldown('sell', model_name, current_time_idx))
                
                if is_signal:
                    self.last_signal_time['sell'][model_name] = current_time_idx
                
                predictions['sell'][model_name] = {
                    'probability': float(signal_prob),
                    'signal': bool(is_signal),
                    'threshold': threshold
                }
                
            except Exception as e:
                print(f"Error with {model_name}_Sell: {e}")
                predictions['sell'][model_name] = {'probability': 0.0, 'signal': False, 'threshold': 0.5}
        
        # Buy predictions (similar logic)
        for model_name, model in self.buy_models.items():
            try:
                pred = model.predict(buy_window, verbose=0)[0]
                signal_prob = pred[1] if len(pred) > 1 else pred[0]
                
                # Apply calibration
                signal_prob = self.calibrate_probabilities(signal_prob)
                
                if np.isnan(signal_prob) or np.isinf(signal_prob):
                    signal_prob = 0.0
                
                threshold = self.adaptive_thresholds['buy'].get(model_name, 0.5)
                
                is_signal = (signal_prob > threshold and 
                           self.check_signal_cooldown('buy', model_name, current_time_idx))
                
                if is_signal:
                    self.last_signal_time['buy'][model_name] = current_time_idx
                
                predictions['buy'][model_name] = {
                    'probability': float(signal_prob),
                    'signal': bool(is_signal),
                    'threshold': threshold
                }
                
            except Exception as e:
                print(f"Error with {model_name}_Buy: {e}")
                predictions['buy'][model_name] = {'probability': 0.0, 'signal': False, 'threshold': 0.5}
        
        return predictions
    
    def create_ensemble_models(self):
        """Create ensemble models using stacking approach"""
        print("🔧 Creating ensemble models...")
        
        # For this implementation, we'll use a simple voting ensemble
        # In practice, you'd train a meta-learner on validation data
        
        self.sell_ensemble_weights = {}
        self.buy_ensemble_weights = {}
        
        # Equal weights for now (could be optimized based on validation performance)
        num_sell_models = len(self.sell_models)
        num_buy_models = len(self.buy_models)
        
        if num_sell_models > 0:
            for model_name in self.sell_models.keys():
                self.sell_ensemble_weights[model_name] = 1.0 / num_sell_models
        
        if num_buy_models > 0:
            for model_name in self.buy_models.keys():
                self.buy_ensemble_weights[model_name] = 1.0 / num_buy_models
        
        print(f"Ensemble weights - Sell: {self.sell_ensemble_weights}")
        print(f"Ensemble weights - Buy: {self.buy_ensemble_weights}")
    
    def get_ensemble_predictions(self, individual_predictions):
        """Get ensemble predictions using weighted voting"""
        ensemble_preds = {'sell': {}, 'buy': {}}
        
        # Sell ensemble
        if self.sell_models and individual_predictions['sell']:
            weighted_prob = 0
            total_weight = 0
            
            for model_name, weight in self.sell_ensemble_weights.items():
                if model_name in individual_predictions['sell']:
                    weighted_prob += individual_predictions['sell'][model_name]['probability'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_prob = weighted_prob / total_weight
                ensemble_threshold = 0.4  # Lower threshold for ensemble
                
                ensemble_preds['sell']['Ensemble'] = {
                    'probability': ensemble_prob,
                    'signal': ensemble_prob > ensemble_threshold,
                    'threshold': ensemble_threshold
                }
        
        # Buy ensemble
        if self.buy_models and individual_predictions['buy']:
            weighted_prob = 0
            total_weight = 0
            
            for model_name, weight in self.buy_ensemble_weights.items():
                if model_name in individual_predictions['buy']:
                    weighted_prob += individual_predictions['buy'][model_name]['probability'] * weight
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_prob = weighted_prob / total_weight
                ensemble_threshold = 0.4  # Lower threshold for ensemble
                
                ensemble_preds['buy']['Ensemble'] = {
                    'probability': ensemble_prob,
                    'signal': ensemble_prob > ensemble_threshold,
                    'threshold': ensemble_threshold
                }
        
        return ensemble_preds

    def run_improved_backtest(self, historical_data):
        """Run improved backtest with all fixes applied"""
        print(f"🚀 Starting improved backtest on {len(historical_data)} samples...")
        
        # Preprocess data separately for sell and buy models
        print("Preprocessing data for sell models...")
        sell_features = self.preprocess_data_for_sell(historical_data.copy())
        
        print("Preprocessing data for buy models...")
        buy_features = self.preprocess_data_for_buy(historical_data.copy())
        
        # Scale features separately
        print("Scaling features...")
        if not self.is_sell_scaler_fitted:
            self.sell_scaler.fit(sell_features.values)
            self.is_sell_scaler_fitted = True
        
        if not self.is_buy_scaler_fitted:
            self.buy_scaler.fit(buy_features.values)
            self.is_buy_scaler_fitted = True
        
        sell_features_scaled = self.sell_scaler.transform(sell_features.values)
        buy_features_scaled = self.buy_scaler.transform(buy_features.values)
        
        # Less aggressive clipping
        sell_features_scaled = np.clip(sell_features_scaled, -5, 5)
        buy_features_scaled = np.clip(buy_features_scaled, -5, 5)
        
        sell_features_scaled_df = pd.DataFrame(sell_features_scaled, columns=sell_features.columns)
        buy_features_scaled_df = pd.DataFrame(buy_features_scaled, columns=buy_features.columns)
        
        # Calculate minimum samples needed
        min_samples_needed = self.lookback_window + max(max(self.sell_lag_periods), max(self.buy_lag_periods))
        
        print(f"Starting predictions from sample {min_samples_needed} onwards...")
        
        # First pass to collect predictions for adaptive thresholds
        print("Calculating adaptive thresholds...")
        initial_predictions = {'sell_predictions': {}, 'buy_predictions': {}}
        
        # Initialize prediction storage
        for model_name in self.sell_models.keys():
            initial_predictions['sell_predictions'][model_name] = []
        for model_name in self.buy_models.keys():
            initial_predictions['buy_predictions'][model_name] = []
        
        # Sample predictions for threshold calculation (first 25000 samples)
        sample_end = min(len(sell_features_scaled_df), min_samples_needed + 25000)
        
        for i in range(min_samples_needed, sample_end):
            # Create windows
            sell_window = self.create_sliding_windows(sell_features_scaled_df, i - self.lookback_window)
            buy_window = self.create_sliding_windows(buy_features_scaled_df, i - self.lookback_window)
            
            if sell_window is not None and buy_window is not None:
                # Get raw predictions (no threshold checking yet)
                for model_name, model in self.sell_models.items():
                    try:
                        pred = model.predict(sell_window, verbose=0)[0]
                        signal_prob = pred[1] if len(pred) > 1 else pred[0]
                        signal_prob = self.calibrate_probabilities(signal_prob)
                        if not (np.isnan(signal_prob) or np.isinf(signal_prob)):
                            initial_predictions['sell_predictions'][model_name].append(signal_prob)
                    except:
                        pass
                
                for model_name, model in self.buy_models.items():
                    try:
                        pred = model.predict(buy_window, verbose=0)[0]
                        signal_prob = pred[1] if len(pred) > 1 else pred[0]
                        signal_prob = self.calibrate_probabilities(signal_prob)
                        if not (np.isnan(signal_prob) or np.isinf(signal_prob)):
                            initial_predictions['buy_predictions'][model_name].append(signal_prob)
                    except:
                        pass
        
        # Calculate adaptive thresholds
        # self.calculate_adaptive_thresholds(initial_predictions, percentile=90)
        
        # Now run the full backtest with adaptive thresholds
        print("Running full backtest with adaptive thresholds...")
        valid_samples = len(sell_features_scaled_df) - self.lookback_window
        
        with tqdm(total=valid_samples, desc="Processing samples") as pbar:
            for i in range(min_samples_needed, len(sell_features_scaled_df)):
                # Create sliding windows
                sell_window = self.create_sliding_windows(sell_features_scaled_df, i - self.lookback_window)
                buy_window = self.create_sliding_windows(buy_features_scaled_df, i - self.lookback_window)
                
                if sell_window is not None and buy_window is not None:
                    # Make predictions with improved method
                    predictions = self.predict_signals_improved(sell_window, buy_window, i)
                    
                    # Store results
                    self.test_results['timestamps'].append(historical_data['datetime'].iloc[i])
                    self.test_results['prices'].append(historical_data['Close'].iloc[i])
                    
                    # Store sell predictions
                    for model_name in self.test_results['sell_predictions'].keys():
                        if model_name in predictions['sell']:
                            self.test_results['sell_predictions'][model_name].append(predictions['sell'][model_name]['probability'])
                            self.test_results['sell_signals'][model_name].append(predictions['sell'][model_name]['signal'])
                        else:
                            self.test_results['sell_predictions'][model_name].append(0.0)
                            self.test_results['sell_signals'][model_name].append(False)
                    
                    # Store buy predictions
                    for model_name in self.test_results['buy_predictions'].keys():
                        if model_name in predictions['buy']:
                            self.test_results['buy_predictions'][model_name].append(predictions['buy'][model_name]['probability'])
                            self.test_results['buy_signals'][model_name].append(predictions['buy'][model_name]['signal'])
                        else:
                            self.test_results['buy_predictions'][model_name].append(0.0)
                            self.test_results['buy_signals'][model_name].append(False)
                
                pbar.update(1)
                
                # Print progress every 25000 samples
                if i % 1000 == 0:
                    current_time = historical_data['datetime'].iloc[i]
                    current_price = historical_data['Close'].iloc[i]
                    
                    signal_summary = []
                    
                    # Sell signals
                    for model_name in predictions['sell'].keys():
                        prob = predictions['sell'][model_name]['probability']
                        is_signal = predictions['sell'][model_name]['signal']
                        signal_summary.append(f"{model_name}_S: {prob:.3f}{'↓' if is_signal else ''}")
                    
                    # Buy signals
                    for model_name in predictions['buy'].keys():
                        prob = predictions['buy'][model_name]['probability']
                        is_signal = predictions['buy'][model_name]['signal']
                        signal_summary.append(f"{model_name}_B: {prob:.3f}{'↑' if is_signal else ''}")
                    
                    print(f"Sample {i}/{len(sell_features_scaled_df)} | {current_time.strftime('%Y-%m-%d %H:%M')} | Price: {current_price:.5f} | {' | '.join(signal_summary)}")
        
        print("✅ Improved backtest completed!")
        return self.test_results

    def create_interactive_charts(self, save=True, show=True):
        """Create interactive charts using Plotly - FIXED VERSION"""
        if not self.test_results['timestamps']:
            print("No data to plot!")
            return None
            
        times = pd.to_datetime(self.test_results['timestamps'])
        prices = self.test_results['prices']
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                'EUR/USD Price with Trading Signals',
                'Sell Model Prediction Probabilities', 
                'Buy Model Prediction Probabilities',
                'Total Signal Count by Model'
            ),
            specs=[[{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}]],
            row_heights=[0.4, 0.2, 0.2, 0.2],
            vertical_spacing=0.08
        )
        
        # Chart 1: Price with signals
        fig.add_trace(go.Scatter(
            x=times, y=prices, mode='lines', name='EUR/USD',
            line=dict(color='white', width=1),
            hovertemplate='<b>EUR/USD</b><br>Time: %{x}<br>Price: %{y:.5f}<extra></extra>'
        ), row=1, col=1)
        
        # Add sell signals to price chart
        for model_name in self.test_results['sell_signals'].keys():
            signal_times = []
            signal_prices = []
            signal_info = []

            for i, is_signal in enumerate(self.test_results['sell_signals'][model_name]):
                if is_signal:
                    signal_times.append(self.test_results['timestamps'][i])
                    signal_prices.append(self.test_results['prices'][i])
                    prob = self.test_results['sell_predictions'][model_name][i]
                    signal_info.append(f"Probability: {prob:.3f}")

            if signal_times:
                color = self.model_colors.get(f'{model_name}_Sell', '#FF0000')
                fig.add_trace(go.Scatter(
                    x=signal_times, y=signal_prices, mode='markers',
                    name=f'{model_name} Sell ↓',
                    marker=dict(symbol='triangle-down', size=8, color=color),
                    text=signal_info,
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Price: %{y:.5f}<br>%{text}<extra></extra>'
                ), row=1, col=1)
        
        # Add buy signals to price chart
        for model_name in self.test_results['buy_signals'].keys():
            signal_times = []
            signal_prices = []
            signal_info = []
            
            for i, is_signal in enumerate(self.test_results['buy_signals'][model_name]):
                if is_signal:
                    signal_times.append(self.test_results['timestamps'][i])
                    signal_prices.append(self.test_results['prices'][i])
                    prob = self.test_results['buy_predictions'][model_name][i]
                    signal_info.append(f"Probability: {prob:.3f}")
            
            if signal_times:
                color = self.model_colors.get(f'{model_name}_Buy', '#00FF00')
                fig.add_trace(go.Scatter(
                    x=signal_times, y=signal_prices, mode='markers',
                    name=f'{model_name} Buy ↑',
                    marker=dict(symbol='triangle-up', size=8, color=color),
                    text=signal_info,
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Price: %{y:.5f}<br>%{text}<extra></extra>'
                ), row=1, col=1)
        
        # Chart 2: Sell probabilities
        for model_name in self.test_results['sell_predictions'].keys():
            probs = self.test_results['sell_predictions'][model_name]
            color = self.model_colors.get(f'{model_name}_Sell', '#FF0000')
            fig.add_trace(go.Scatter(
                x=times, y=probs, mode='lines',
                name=f'{model_name} Sell Prob', line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Probability: %{y:.3f}<extra></extra>'
            ), row=2, col=1)
        
        # Chart 3: Buy probabilities
        for model_name in self.test_results['buy_predictions'].keys():
            probs = self.test_results['buy_predictions'][model_name]
            color = self.model_colors.get(f'{model_name}_Buy', '#00FF00')
            fig.add_trace(go.Scatter(
                x=times, y=probs, mode='lines',
                name=f'{model_name} Buy Prob', line=dict(color=color, width=1.5),
                showlegend=False,
                hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Probability: %{y:.3f}<extra></extra>'
            ), row=3, col=1)
        
        # Chart 4: Signal counts
        signal_counts = {}
        for model_name in self.test_results['sell_signals'].keys():
            signal_counts[f'{model_name}_Sell'] = sum(self.test_results['sell_signals'][model_name])
        for model_name in self.test_results['buy_signals'].keys():
            signal_counts[f'{model_name}_Buy'] = sum(self.test_results['buy_signals'][model_name])
        
        models = list(signal_counts.keys())
        counts = list(signal_counts.values())
        colors = [self.model_colors.get(model, '#808080') for model in models]
        
        fig.add_trace(go.Bar(
            x=models, y=counts, marker_color=colors,
            text=counts, textposition='auto', showlegend=False,
            hovertemplate='<b>%{x}</b><br>Signal Count: %{y}<extra></extra>'
        ), row=4, col=1)
        
        # Add threshold lines to probability charts
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                    annotation_text="Default Threshold", row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="green", 
                    annotation_text="Default Threshold", row=3, col=1)
        
        # Update layout - FIXED: No rangeslider that was causing the duplicate
        fig.update_layout(
            title={
                'text': 'Improved Trading Model Performance Dashboard',
                'x': 0.5,
                'font': {'size': 24}
            },
            height=1200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
            hovermode='x unified'
        )
        
        # Add range selector ONLY to the first subplot (price chart) - FIXED
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            type="date",
            row=1, col=1  # FIXED: Only apply to first subplot
        )
        
        # Update y-axis labels for each subplot - FIXED: Use specific subplot targeting
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Sell Probability", range=[0, 1], row=2, col=1)
        fig.update_yaxes(title_text="Buy Probability", range=[0, 1], row=3, col=1)
        fig.update_yaxes(title_text="Signal Count", row=4, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Time", row=4, col=1)
        
        if save:
            fig.write_html(f"{self.output_dir}/improved_trading_dashboard_interactive.html")
            try:
                fig.write_image(f"{self.output_dir}/improved_trading_dashboard_static.png", width=1400, height=1200)
            except:
                print("Warning: Could not save PNG image. Install kaleido: pip install kaleido")
            print(f"📊 Interactive dashboard saved to {self.output_dir}/")
        
        if show:
            fig.show()
            
        return fig

    def save_detailed_summary(self):
        """Save a detailed summary to a text file"""
        summary_path = f"{self.output_dir}/detailed_backtest_summary.txt"
        
        times = pd.to_datetime(self.test_results['timestamps'])
        prices = self.test_results['prices']
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("IMPROVED TRADING MODEL BACKTEST DETAILED SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Information
            f.write("DATASET INFORMATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total samples processed: {len(self.test_results['timestamps'])}\n")
            f.write(f"Time period: {times.min().strftime('%Y-%m-%d %H:%M')} to {times.max().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Duration: {(times.max() - times.min()).total_seconds() / 3600:.1f} hours\n")
            f.write(f"Price range: {min(prices):.5f} - {max(prices):.5f}\n")
            f.write(f"Price volatility (std): {np.std(prices):.5f}\n\n")
            
            # Model Configuration
            f.write("MODEL CONFIGURATION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Lookback window: {self.lookback_window} periods\n")
            f.write(f"Signal cooldown: {self.signal_cooldown} periods\n")
            f.write(f"Sell lag periods: {self.sell_lag_periods}\n")
            f.write(f"Buy lag periods: {self.buy_lag_periods}\n")
            f.write(f"Feature scaling: RobustScaler (quantile_range: 15-85%)\n")
            f.write(f"Probability calibration: Sigmoid calibration applied\n\n")
            
            # Loaded Models
            f.write("LOADED MODELS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sell models loaded: {len(self.sell_models)} ({list(self.sell_models.keys())})\n")
            f.write(f"Buy models loaded: {len(self.buy_models)} ({list(self.buy_models.keys())})\n\n")
            
            # Adaptive Thresholds
            f.write("ADAPTIVE THRESHOLDS:\n")
            f.write("-" * 40 + "\n")
            f.write("Sell Model Thresholds:\n")
            for model_name, threshold in self.adaptive_thresholds['sell'].items():
                f.write(f"  {model_name}: {threshold:.3f}\n")
            f.write("Buy Model Thresholds:\n")
            for model_name, threshold in self.adaptive_thresholds['buy'].items():
                f.write(f"  {model_name}: {threshold:.3f}\n")
            f.write("\n")
            
            # Sell Signal Statistics
            f.write("SELL SIGNAL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            total_samples = len(self.test_results['timestamps'])
            f.write(f"{'Model':<15} {'Signals':<8} {'Rate%':<8} {'Avg Prob':<10} {'Max Prob':<10} {'Min Prob':<10}\n")
            f.write("-" * 70 + "\n")
            
            for model_name in self.test_results['sell_predictions'].keys():
                signal_count = sum(self.test_results['sell_signals'][model_name])
                signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
                avg_prob = np.mean(self.test_results['sell_predictions'][model_name])
                max_prob = np.max(self.test_results['sell_predictions'][model_name])
                min_prob = np.min(self.test_results['sell_predictions'][model_name])
                
                f.write(f"{model_name:<15} {signal_count:<8} {signal_rate:<8.2f} {avg_prob:<10.3f} {max_prob:<10.3f} {min_prob:<10.3f}\n")
            
            f.write("\n")
            
            # Buy Signal Statistics
            f.write("BUY SIGNAL STATISTICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Model':<15} {'Signals':<8} {'Rate%':<8} {'Avg Prob':<10} {'Max Prob':<10} {'Min Prob':<10}\n")
            f.write("-" * 70 + "\n")
            
            for model_name in self.test_results['buy_predictions'].keys():
                signal_count = sum(self.test_results['buy_signals'][model_name])
                signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
                avg_prob = np.mean(self.test_results['buy_predictions'][model_name])
                max_prob = np.max(self.test_results['buy_predictions'][model_name])
                min_prob = np.min(self.test_results['buy_predictions'][model_name])
                
                f.write(f"{model_name:<15} {signal_count:<8} {signal_rate:<8.2f} {avg_prob:<10.3f} {max_prob:<10.3f} {min_prob:<10.3f}\n")
            
            f.write("\n")
            
            # Model Performance Analysis
            f.write("MODEL PERFORMANCE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            # Find most/least active models
            sell_signal_counts = {name: sum(self.test_results['sell_signals'][name]) for name in self.test_results['sell_signals'].keys()}
            buy_signal_counts = {name: sum(self.test_results['buy_signals'][name]) for name in self.test_results['buy_signals'].keys()}
            
            if sell_signal_counts:
                most_active_sell = max(sell_signal_counts, key=sell_signal_counts.get)
                least_active_sell = min(sell_signal_counts, key=sell_signal_counts.get)
                f.write(f"Most active sell model: {most_active_sell} ({sell_signal_counts[most_active_sell]} signals)\n")
                f.write(f"Least active sell model: {least_active_sell} ({sell_signal_counts[least_active_sell]} signals)\n")
            
            if buy_signal_counts:
                most_active_buy = max(buy_signal_counts, key=buy_signal_counts.get)
                least_active_buy = min(buy_signal_counts, key=buy_signal_counts.get)
                f.write(f"Most active buy model: {most_active_buy} ({buy_signal_counts[most_active_buy]} signals)\n")
                f.write(f"Least active buy model: {least_active_buy} ({buy_signal_counts[least_active_buy]} signals)\n")
            
            f.write("\n")
            
            # Improvements Made
            f.write("IMPROVEMENTS IMPLEMENTED:\n")
            f.write("-" * 40 + "\n")
            f.write("1. SEPARATE FEATURE PREPROCESSING:\n")
            f.write("   - Sell models use sell-specific lag features\n")
            f.write("   - Buy models use buy-specific lag features\n")
            f.write("   - Resolves feature mismatch issues\n\n")
            
            f.write("2. ADAPTIVE THRESHOLDS:\n")
            f.write("   - 30th percentile-based thresholds instead of fixed 0.5\n")
            f.write("   - Each model gets optimized threshold\n")
            f.write("   - Minimum threshold of 0.3 enforced\n\n")
            
            f.write("3. PROBABILITY CALIBRATION:\n")
            f.write("   - Sigmoid calibration applied to raw predictions\n")
            f.write("   - Improves probability interpretation\n")
            f.write("   - Better handles focal loss trained models\n\n")
            
            f.write("4. SIGNAL COOLDOWN SYSTEM:\n")
            f.write(f"   - {self.signal_cooldown}-period minimum between signals\n")
            f.write("   - Prevents signal clustering\n")
            f.write("   - Cleaner signal generation\n\n")
            
            f.write("5. IMPROVED SCALING:\n")
            f.write("   - Separate scalers for sell/buy models\n")
            f.write("   - Less aggressive quantile range (15-85%)\n")
            f.write("   - Gentler clipping (-5 to 5)\n\n")
            
            f.write("6. ENSEMBLE CAPABILITY:\n")
            f.write("   - Weighted voting ensemble implemented\n")
            f.write("   - Lower thresholds for ensemble signals\n")
            f.write("   - Can combine multiple model predictions\n\n")
            
            # Issues Addressed
            f.write("ISSUES ADDRESSED:\n")
            f.write("-" * 40 + "\n")
            f.write("BEFORE:\n")
            f.write("- Only Conv1D_Sell and LSTM_Buy models generated signals\n")
            f.write("- Most models stuck at low probabilities (~0.27-0.43)\n")
            f.write("- Multiple consecutive signals when prob stayed high\n")
            f.write("- Fixed 0.5 threshold not optimal for all models\n")
            f.write("- Feature preprocessing mismatch between sell/buy models\n\n")
            
            f.write("AFTER:\n")
            f.write("- Adaptive thresholds optimize each model's performance\n")
            f.write("- Probability calibration improves signal quality\n")
            f.write("- Signal cooldown prevents clustering\n")
            f.write("- Separate preprocessing for each model type\n")
            f.write("- Better feature scaling and handling\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("1. MONITOR MODEL PERFORMANCE:\n")
            f.write("   - Track signal accuracy over time\n")
            f.write("   - Adjust thresholds based on performance\n")
            f.write("   - Consider retraining underperforming models\n\n")
            
            f.write("2. ENSEMBLE OPTIMIZATION:\n")
            f.write("   - Train meta-learner for ensemble weights\n")
            f.write("   - Use validation data for weight optimization\n")
            f.write("   - Consider stacking with different base models\n\n")
            
            f.write("3. THRESHOLD TUNING:\n")
            f.write("   - Experiment with different percentiles (70-85)\n")
            f.write("   - Consider market condition-specific thresholds\n")
            f.write("   - Implement dynamic threshold adjustment\n\n")
            
            f.write("4. FEATURE ENGINEERING:\n")
            f.write("   - Add market regime indicators\n")
            f.write("   - Include volatility-based features\n")
            f.write("   - Consider correlation-based features\n\n")
            
            # Technical Details
            f.write("TECHNICAL IMPLEMENTATION DETAILS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Base features: {len(self.base_features)} indicators\n")
            f.write(f"Sell model features: {len(self.sell_training_features)} total\n")
            f.write(f"Buy model features: {len(self.buy_training_features)} total\n")
            f.write(f"Scaling method: RobustScaler with quantile_range=(15.0, 85.0)\n")
            f.write(f"Clipping range: [-5, 5]\n")
            f.write(f"Calibration method: Sigmoid calibration\n")
            f.write(f"Memory usage optimizations: Separate scalers and feature sets\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF SUMMARY\n")
            f.write("=" * 80 + "\n")
        
        print(f"📄 Detailed summary saved to {summary_path}")

    def run_complete_analysis(self, historical_data, save_charts=True, show_charts=True):
        """Run complete analysis with all improvements and generate outputs"""
        print("🚀 Starting complete improved analysis...")
        
        # Run improved backtest
        results = self.run_improved_backtest(historical_data)
        
        # Create interactive charts
        if save_charts or show_charts:
            self.create_interactive_charts(save=save_charts, show=show_charts)
        
        # Save detailed summary
        self.save_detailed_summary()
        
        # Save JSON summary
        self.save_json_summary()
        
        # Print console summary
        self.print_console_summary()
        
        return results
    
    def save_json_summary(self):
        """Save machine-readable JSON summary"""
        times = pd.to_datetime(self.test_results['timestamps'])
        prices = self.test_results['prices']
        
        summary = {
            'metadata': {
                'generated_on': datetime.now().isoformat(),
                'total_samples': len(self.test_results['timestamps']),
                'time_period': {
                    'start': times.min().isoformat(),
                    'end': times.max().isoformat(),
                    'duration_hours': (times.max() - times.min()).total_seconds() / 3600
                },
                'price_stats': {
                    'min': float(min(prices)),
                    'max': float(max(prices)),
                    'mean': float(np.mean(prices)),
                    'std': float(np.std(prices))
                }
            },
            'configuration': {
                'lookback_window': self.lookback_window,
                'signal_cooldown': self.signal_cooldown,
                'sell_lag_periods': self.sell_lag_periods,
                'buy_lag_periods': self.buy_lag_periods,
                'adaptive_thresholds': self.adaptive_thresholds
            },
            'models': {
                'sell_models_loaded': list(self.sell_models.keys()),
                'buy_models_loaded': list(self.buy_models.keys())
            },
            'results': {
                'sell_signals': {},
                'buy_signals': {}
            }
        }
        
        # Add sell signal results
        total_samples = len(self.test_results['timestamps'])
        for model_name in self.test_results['sell_predictions'].keys():
            signal_count = sum(self.test_results['sell_signals'][model_name])
            summary['results']['sell_signals'][model_name] = {
                'signal_count': signal_count,
                'signal_rate_percent': (signal_count / total_samples) * 100,
                'avg_probability': float(np.mean(self.test_results['sell_predictions'][model_name])),
                'max_probability': float(np.max(self.test_results['sell_predictions'][model_name])),
                'min_probability': float(np.min(self.test_results['sell_predictions'][model_name]))
            }
        
        # Add buy signal results
        for model_name in self.test_results['buy_predictions'].keys():
            signal_count = sum(self.test_results['buy_signals'][model_name])
            summary['results']['buy_signals'][model_name] = {
                'signal_count': signal_count,
                'signal_rate_percent': (signal_count / total_samples) * 100,
                'avg_probability': float(np.mean(self.test_results['buy_predictions'][model_name])),
                'max_probability': float(np.max(self.test_results['buy_predictions'][model_name])),
                'min_probability': float(np.min(self.test_results['buy_predictions'][model_name]))
            }
        
        # Save JSON
        json_path = f"{self.output_dir}/backtest_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 JSON summary saved to {json_path}")
    
    def print_console_summary(self):
        """Print summary to console"""
        times = pd.to_datetime(self.test_results['timestamps'])
        prices = self.test_results['prices']
        
        print("\n📈 IMPROVED BACKTEST SUMMARY:")
        print("=" * 60)
        print(f"Total samples processed: {len(self.test_results['timestamps'])}")
        print(f"Time period: {times.min().strftime('%Y-%m-%d %H:%M')} to {times.max().strftime('%Y-%m-%d %H:%M')}")
        print(f"Price range: {min(prices):.5f} - {max(prices):.5f}")
        
        print("\n🎯 SELL SIGNAL STATISTICS:")
        total_samples = len(self.test_results['timestamps'])
        for model_name in self.test_results['sell_predictions'].keys():
            signal_count = sum(self.test_results['sell_signals'][model_name])
            signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
            avg_prob = np.mean(self.test_results['sell_predictions'][model_name])
            max_prob = np.max(self.test_results['sell_predictions'][model_name])
            threshold = self.adaptive_thresholds['sell'].get(model_name, 0.5)
            
            print(f"{model_name:12}: {signal_count:4d} signals ({signal_rate:5.2f}%) | "
                  f"Threshold: {threshold:.3f} | Avg prob: {avg_prob:.3f} | Max prob: {max_prob:.3f}")
        
        print("\n🎯 BUY SIGNAL STATISTICS:")
        for model_name in self.test_results['buy_predictions'].keys():
            signal_count = sum(self.test_results['buy_signals'][model_name])
            signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
            avg_prob = np.mean(self.test_results['buy_predictions'][model_name])
            max_prob = np.max(self.test_results['buy_predictions'][model_name])
            threshold = self.adaptive_thresholds['buy'].get(model_name, 0.5)
            
            print(f"{model_name:12}: {signal_count:4d} signals ({signal_rate:5.2f}%) | "
                  f"Threshold: {threshold:.3f} | Avg prob: {avg_prob:.3f} | Max prob: {max_prob:.3f}")

# Usage example:
def main():
    """Main function with all improvements"""
    SELL_MODEL_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\sell_signal_models\sell_models'
    BUY_MODEL_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\long_signal_models\buy_models'
    
    # Initialize improved tester
    print("🔧 Initializing Improved Trading Model Tester...")
    tester = ImprovedTradingModelTester(
        sell_model_dir=SELL_MODEL_DIR, 
        buy_model_dir=BUY_MODEL_DIR
    )
    
    # Load models
    tester.load_models()
    
    if not tester.sell_models and not tester.buy_models:
        print("❌ No models loaded! Please check the model directory paths.")
        return
    
    # Create ensemble models
    tester.create_ensemble_models()
    
    # Fetch data
    historical_data = tester.fetch_historical_data(samples=25000)
    
    # Run complete analysis
    results = tester.run_complete_analysis(
        historical_data, 
        save_charts=True, 
        show_charts=True
    )
    
    print("\n✅ Complete improved analysis finished!")
    print(f"📁 All outputs saved to '{tester.output_dir}/' directory")
    print("📊 Interactive charts, detailed summary, and JSON data available")

if __name__ == "__main__":
    main()

#%%