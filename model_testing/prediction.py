import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import yfinance as yf
import ta  # Using 'ta' library instead of talib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import time
import os
from scipy import stats

# Configure matplotlib for better performance
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'

class LiveTradingPredictor:
    def __init__(self, model_dir='sell_models', lookback_window=120):
        self.model_dir = model_dir
        self.lookback_window = lookback_window
        self.lag_periods = [70, 24, 10, 74, 39]
        self.models = {}
        self.model_colors = {
            'LSTM': 'red',
            'GRU': 'orange', 
            'Conv1D': 'yellow',
            'Conv1D_LSTM': 'cyan'
        }
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.data_buffer = pd.DataFrame()
        self.signals_history = {name: [] for name in self.model_colors.keys()}
        self.is_scaler_fitted = False
        
        # Define the exact feature list used during training (from your column list)
        self.training_features = [
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
        
        # Add lag features to the training features list
        for lag in self.lag_periods:
            self.training_features.append(f'short_signal_lag_{lag}')
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                self.training_features.append(f'{indicator}_lag_{lag}')
        
        # Add rolling stats on lag features
        self.training_features.extend(['target_lag_mean', 'target_lag_std'])
        
        # Add volume transformations
        self.training_features.extend(['Volume_log', 'Volume_winsor', 'Volume_rank'])
        
        print(f"Expected {len(self.training_features)} features for model compatibility")
        
        # Load models
        self.load_models()
        
        # Setup plot
        self.setup_plot()
        
    def load_models(self):
        """Load all trained models"""
        print("Loading trained models...")
        model_files = {
            'LSTM': 'LSTM.h5',
            'GRU': 'GRU.h5', 
            'Conv1D': 'Conv1D.h5',
            'Conv1D_LSTM': 'Conv1D_LSTM.h5'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects={'focal_loss_fn': self.focal_loss_fn}
                    )
                    
                    # Check model input shape
                    expected_shape = model.input_shape
                    print(f"✅ Loaded {model_name} model - Expected input shape: {expected_shape}")
                    
                    # Extract expected number of features
                    if len(expected_shape) == 3:  # (batch_size, timesteps, features)
                        expected_features = expected_shape[2]
                        if expected_features != len(self.training_features):
                            print(f"⚠️  Shape mismatch for {model_name}: expected {expected_features}, got {len(self.training_features)}")
                            # Adjust training features list to match model expectation
                            if expected_features < len(self.training_features):
                                self.training_features = self.training_features[:expected_features]
                                print(f"   Trimmed feature list to {len(self.training_features)} features")
                            elif expected_features > len(self.training_features):
                                # Pad with dummy features
                                additional_features = expected_features - len(self.training_features)
                                for i in range(additional_features):
                                    self.training_features.append(f'dummy_feature_{i}')
                                print(f"   Added {additional_features} dummy features")
                    
                    self.models[model_name] = model
                    
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
            else:
                print(f"❌ Model file not found: {model_path}")
        
        print(f"Final feature count: {len(self.training_features)}")
    
    def focal_loss_fn(self, y_true, y_pred):
        """Focal loss function for model loading"""
        gamma = 2.0
        alpha = 0.25
        epsilon = 1e-5
        
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        
        loss = -alpha_t * tf.pow(1. - pt, gamma) * tf.math.log(pt + epsilon)
        loss = tf.where(tf.math.is_finite(loss), loss, tf.zeros_like(loss))
        
        return tf.reduce_mean(loss)
        
    def fetch_live_data(self, period="1d", interval="1m"):
        """Fetch live EUR/USD data"""
        try:
            ticker = yf.Ticker("EURUSD=X")
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                print("No data received from yfinance, using sample data...")
                return self.generate_sample_data()
                
            # Reset index to get datetime as column
            data = data.reset_index()
            
            # Handle variable number of columns from yfinance
            print(f"Received {len(data.columns)} columns from yfinance: {list(data.columns)}")
            
            # Map columns correctly regardless of what yfinance returns
            required_columns = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
            new_data = pd.DataFrame()
            
            # First column should be datetime/timestamp
            new_data['datetime'] = data.iloc[:, 0]
            
            # Map OHLCV columns based on what's available
            column_mapping = {}
            available_cols = [col.lower() for col in data.columns]
            
            for i, col in enumerate(data.columns):
                col_lower = col.lower()
                if 'open' in col_lower:
                    column_mapping['Open'] = col
                elif 'high' in col_lower:
                    column_mapping['High'] = col
                elif 'low' in col_lower:
                    column_mapping['Low'] = col
                elif 'close' in col_lower:
                    column_mapping['Close'] = col
                elif 'volume' in col_lower:
                    column_mapping['Volume'] = col
            
            # Extract OHLCV data
            for target_col in ['Open', 'High', 'Low', 'Close']:
                if target_col in column_mapping:
                    new_data[target_col] = data[column_mapping[target_col]]
                else:
                    print(f"Warning: {target_col} column not found, using Close price")
                    new_data[target_col] = data[column_mapping.get('Close', data.columns[1])]
            
            # Handle Volume (forex might not have volume)
            if 'Volume' in column_mapping:
                new_data['Volume'] = data[column_mapping['Volume']]
            else:
                print("Volume not available, generating synthetic volume")
                new_data['Volume'] = np.random.randint(1000, 10000, len(new_data))
            
            # Add Date and Time columns
            new_data['Date'] = new_data['datetime'].dt.strftime('%Y%m%d').astype(int)
            new_data['Time'] = new_data['datetime'].dt.strftime('%H:%M:%S')
            
            print(f"Successfully processed data: {len(new_data)} rows, {len(new_data.columns)} columns")
            return new_data
            
        except Exception as e:
            print(f"Error fetching live data: {e}")
            print("Falling back to sample data...")
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        """Generate sample EUR/USD data for testing"""
        print("Generating sample data for testing...")
        dates = pd.date_range(start=datetime.now() - timedelta(days=1), 
                             end=datetime.now(), freq='1min')
        
        # Generate realistic EUR/USD price movement
        np.random.seed(42)
        base_price = 1.0850
        returns = np.random.normal(0, 0.0001, len(dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame({
            'datetime': dates,
            'Open': prices,
            'High': prices * np.random.uniform(1.0, 1.002, len(dates)),
            'Low': prices * np.random.uniform(0.998, 1.0, len(dates)),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        })
        
        # Ensure High >= Close >= Low and High >= Open >= Low
        for i in range(len(data)):
            data.loc[i, 'High'] = max(data.loc[i, 'High'], data.loc[i, 'Open'], data.loc[i, 'Close'])
            data.loc[i, 'Low'] = min(data.loc[i, 'Low'], data.loc[i, 'Open'], data.loc[i, 'Close'])
        
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

            # Add custom STOCH_K calculation if missing
            if 'STOCHF_K' not in data.columns:
                # Calculate STOCH_K manually (using 14-period lookback)
                low_min = data['Low'].rolling(window=14).min()
                high_max = data['High'].rolling(window=14).max()
                data['STOCHF_K'] = 100 * (data['Close'] - low_min) / (high_max - low_min)
            
            # If 'STOCHF_K' is calculated, rename it as 'STOCH_K' for consistency
            if 'STOCHF_K' in data.columns:
                data['STOCH_K'] = data['STOCHF_K']
                data = data.drop(columns=['STOCHF_K'])  # Drop temporary 'STOCHF_K'
                
            # Map ta column names to expected names from training
            column_mapping = {
                # Volume indicators
                'volume_adi': 'AD',
                'volume_obv': 'OBV',
                'volume_mfi': 'MFI',
                
                # Volatility indicators  
                'volatility_atr': 'ATR',
                'volatility_bbh': 'BBH',
                'volatility_bbl': 'BBL',
                'volatility_bbm': 'BBM',
                'volatility_bbhi': 'BBHI',
                'volatility_bbli': 'BBLI',
                'volatility_kcc': 'KCC',
                'volatility_kch': 'KCH',
                'volatility_kcl': 'KCL',
                'volatility_kchi': 'KCHI',
                'volatility_kcli': 'KCLI',
                'volatility_dcl': 'DCL',
                'volatility_dch': 'DCH',
                'volatility_dcm': 'DCM',
                'volatility_dchi': 'DCHI',
                'volatility_dcli': 'DCLI',
                'volatility_ui': 'UI',
                
                # Trend indicators
                'trend_macd': 'MACD',
                'trend_macd_signal': 'MACDSIGNAL', 
                'trend_macd_diff': 'MACDHIST',
                'trend_adx': 'ADX',
                'trend_adx_pos': 'PLUS_DI',
                'trend_adx_neg': 'MINUS_DI',
                'trend_vortex_ind_pos': 'PLUS_VI',
                'trend_vortex_ind_neg': 'MINUS_VI',
                'trend_trix': 'TRIX',
                'trend_mass_index': 'MASS_INDEX',
                'trend_dpo': 'DPO',
                'trend_kst': 'KST',
                'trend_kst_sig': 'KST_SIG',
                'trend_kst_diff': 'KST_DIFF',
                'trend_ichimoku_conv': 'ICHIMOKU_CONV',
                'trend_ichimoku_base': 'ICHIMOKU_BASE',
                'trend_ichimoku_a': 'ICHIMOKU_A',
                'trend_ichimoku_b': 'ICHIMOKU_B',
                'trend_aroon_up': 'AROON_UP',
                'trend_aroon_down': 'AROON_DOWN',
                'trend_aroon_ind': 'AROONOSC',
                'trend_psar_up': 'PSAR_UP',
                'trend_psar_down': 'PSAR_DOWN',
                'trend_psar_up_indicator': 'PSAR_UP_IND',
                'trend_psar_down_indicator': 'PSAR_DOWN_IND',
                
                # Momentum indicators
                'momentum_rsi': 'RSI',
                'momentum_stoch_rsi': 'STOCHRSI',
                'momentum_stoch_rsi_k': 'STOCHRSI_K',
                'momentum_stoch_rsi_d': 'STOCHRSI_D',
                'momentum_tsi': 'TSI',
                'momentum_uo': 'ULTOSC',
                'momentum_stoch': 'STOCH',
                'momentum_stoch_signal': 'STOCH_D',
                'momentum_wr': 'WILLR',
                'momentum_ao': 'AO',
                'momentum_kama': 'KAMA',
                'momentum_roc': 'ROC',
                'momentum_ppo': 'PPO',
                'momentum_ppo_signal': 'PPO_SIGNAL',
                'momentum_ppo_hist': 'PPO_HIST',
                
                # Others indicators
                'others_dr': 'DR',
                'others_dlr': 'DLR',
                'others_cr': 'CR',
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
            
            # STOCH_K (from momentum_stoch)
            if 'STOCH_K' not in data.columns and 'momentum_stoch' in data.columns:
                data['STOCH_K'] = data['momentum_stoch']
            
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
            required_indicators = [
                'ADOSC', 'ADXR', 'AROONOSC', 'AROON_DOWN', 'AROON_UP', 'BOP', 'CCI',
                'CMO', 'EFI', 'HT_DCPERIOD', 'HT_DCPHASE', 'HT_LEADSINE', 'HT_PHASOR_INPHASE',
                'HT_PHASOR_QUADRATURE', 'HT_SINE', 'HT_TRENDMODE', 'MOM', 'OBV', 'ROCP',
                'ROCR100', 'RSI', 'STOCHF_K', 'STOCHRSI_D', 'STOCHRSI_K', 'TRANGE', 'TRIX',
                'z_score', 'AD', 'ADX', 'APO', 'ATR', 'DX', 'MACD', 'MACDHIST', 'MACDSIGNAL',
                'MFI', 'MINUS_DI', 'MINUS_DM', 'NATR', 'PLUS_DI', 'PLUS_DM', 'PPO', 'ROC',
                'ROCR', 'STOCHF_D', 'STOCH_D', 'STOCH_K', 'ULTOSC', 'VWAP', 'WILLR', 'rolling_std'
            ]
            
            for indicator in required_indicators:
                if indicator not in data.columns:
                    data[indicator] = 0
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        data = data.replace([np.inf, -np.inf], 0)
        
        return data
    
    def add_lag_features(self, data):
        """Add lag features as used in training"""
        print("Adding lag features...")
        
        # Create dummy target column for lag calculation
        data['short_signal'] = 0
        
        for lag in self.lag_periods:
            data[f'short_signal_lag_{lag}'] = data['short_signal'].shift(lag)
            
            # Add lagged versions of key technical indicators
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                if indicator in data.columns:
                    data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
        
        # Add rolling stats on lagged features
        lag_cols = [f'short_signal_lag_{lag}' for lag in self.lag_periods]
        data['target_lag_mean'] = data[lag_cols].mean(axis=1)
        data['target_lag_std'] = data[lag_cols].std(axis=1)
        
        # Remove the dummy target column
        data = data.drop('short_signal', axis=1)
        
        # Handle missing values from lags
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data
    
    def preprocess_data(self, data):
        """Apply the same preprocessing as used in training"""
        print("Preprocessing data...")
        
        # Add technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Add lag features
        data = self.add_lag_features(data)
        
        # Transform volume-based features
        volume_cols = ['Volume']
        for col in volume_cols:
            if col in data.columns:
                data[f'{col}_log'] = np.log1p(data[col])
                q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
                data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
                data[f'{col}_rank'] = data[col].rank(pct=True)
        
        # Select ONLY the features that were used during training, in the exact order
        print(f"Available columns: {len(data.columns)}")
        print(f"Required features: {len(self.training_features)}")
        
        # Create feature dataframe with exact training features
        features = pd.DataFrame()
        missing_features = []
        
        for feature in self.training_features:
            if feature in data.columns:
                features[feature] = data[feature]
            elif feature.startswith('dummy_feature_'):
                # Handle dummy features added for shape matching
                features[feature] = 0
            else:
                # Create missing feature with zeros or intelligent defaults
                if 'lag_' in feature:
                    # For lag features, use the base indicator if available
                    base_indicator = feature.split('_lag_')[0]
                    if base_indicator in data.columns:
                        features[feature] = data[base_indicator].shift(int(feature.split('_lag_')[1]))
                    else:
                        features[feature] = 0
                elif feature in ['target_lag_mean', 'target_lag_std']:
                    # For rolling stats on lag features, calculate if possible
                    features[feature] = 0
                else:
                    # Default to zero for other missing features
                    features[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing, filled with zeros: {missing_features[:10]}...")
        
        # Ensure all features are numeric
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except:
                    features[col] = 0
        
        # Fill any remaining missing values
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        print(f"Final feature shape: {features.shape} (expected: ({len(data)}, {len(self.training_features)}))")
        
        return features, data
    
    def create_sliding_windows(self, features):
        """Create sliding windows for prediction"""
        if len(features) < self.lookback_window:
            print(f"Not enough data for sliding windows. Need {self.lookback_window}, got {len(features)}")
            return None
        
        # Convert to numpy array
        features_array = features.values.astype(np.float32)
        
        # Create sliding window for the most recent data point
        window = features_array[-self.lookback_window:].reshape(1, self.lookback_window, features_array.shape[1])
        
        return window
    
    def scale_features(self, features):
        """Scale features using RobustScaler"""
        if not self.is_scaler_fitted and len(features) >= self.lookback_window:
            print("Fitting scaler on initial data...")
            self.scaler.fit(features.values)
            self.is_scaler_fitted = True
        
        if self.is_scaler_fitted:
            # Scale the features
            features_scaled = self.scaler.transform(features.values)
            features_scaled = np.clip(features_scaled, -10, 10)  # Cap values
            return features_scaled
        else:
            return features.values
    
    def predict_signals(self, window):
        """Make predictions using all loaded models"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Check window shape compatibility (only print once during debugging)
                expected_shape = model.input_shape
                actual_shape = window.shape
                
                # If there's a shape mismatch, try to fix it
                if len(expected_shape) == 3 and len(actual_shape) == 3:
                    if expected_shape[2] != actual_shape[2]:  # Feature dimension mismatch
                        if expected_shape[2] < actual_shape[2]:
                            # Trim features
                            window_adjusted = window[:, :, :expected_shape[2]]
                            print(f"   Trimmed window features from {actual_shape[2]} to {expected_shape[2]}")
                        else:
                            # Pad features with zeros
                            padding_needed = expected_shape[2] - actual_shape[2]
                            padding = np.zeros((actual_shape[0], actual_shape[1], padding_needed))
                            window_adjusted = np.concatenate([window, padding], axis=2)
                            print(f"   Padded window features from {actual_shape[2]} to {expected_shape[2]}")
                        window = window_adjusted
                
                # Make prediction
                pred = model.predict(window, verbose=0)[0]
                
                # Convert to probability/signal strength
                if len(pred) > 1:
                    signal_prob = pred[1]  # Probability of class 1 (short signal)
                else:
                    signal_prob = pred[0]
                
                # Handle potential NaN or inf values
                if np.isnan(signal_prob) or np.isinf(signal_prob):
                    signal_prob = 0.0
                
                # Consider it a signal if probability > 0.7
                is_signal = signal_prob > 0.5
                
                predictions[model_name] = {
                    'probability': float(signal_prob),
                    'signal': bool(is_signal)
                }
                
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = {'probability': 0.0, 'signal': False}
        
        return predictions
    
    def setup_plot(self):
        """Setup the live plotting"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10), 
                                                     gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        self.ax1.set_title('EUR/USD Live Trading Signals', fontsize=16, color='white')
        self.ax1.set_ylabel('Price', fontsize=12, color='white')
        self.ax1.grid(True, alpha=0.3)
        
        # Signals chart
        self.ax2.set_title('Model Signals', fontsize=14, color='white')
        self.ax2.set_ylabel('Signal Strength', fontsize=12, color='white')
        self.ax2.set_xlabel('Time', fontsize=12, color='white')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 1)
        
        # Initialize empty lines
        self.price_line, = self.ax1.plot([], [], 'white', linewidth=2, label='EUR/USD')
        
        # Signal lines for each model
        self.signal_lines = {}
        for model_name, color in self.model_colors.items():
            line, = self.ax2.plot([], [], color=color, linewidth=2, label=f'{model_name}')
            self.signal_lines[model_name] = line
        
        # Legends
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
    def update_plot(self, frame):
        """Update the live plot"""
        try:
            # Fetch new data
            new_data = self.fetch_live_data(period="1d", interval="1m")
            
            if new_data is None or len(new_data) == 0:
                print("No data available, retrying next update...")
                return self.price_line, *self.signal_lines.values()
            
            # Keep only recent data to avoid memory issues
            if len(new_data) > 500:
                new_data = new_data.tail(500)
            
            # Update data buffer
            self.data_buffer = new_data.copy()
            
            # Ensure we have enough data for prediction
            min_data_needed = self.lookback_window + max(self.lag_periods)
            if len(self.data_buffer) < min_data_needed:
                print(f"Waiting for more data... ({len(self.data_buffer)}/{min_data_needed} rows)")
                return self.price_line, *self.signal_lines.values()
            
            # Preprocess data
            try:
                features, processed_data = self.preprocess_data(self.data_buffer)
            except Exception as e:
                print(f"Error in preprocessing: {e}")
                return self.price_line, *self.signal_lines.values()
            
            # Scale features
            try:
                features_scaled = self.scale_features(features)
            except Exception as e:
                print(f"Error in scaling: {e}")
                return self.price_line, *self.signal_lines.values()
            
            # Create sliding windows
            try:
                window = self.create_sliding_windows(pd.DataFrame(features_scaled))
            except Exception as e:
                print(f"Error creating windows: {e}")
                return self.price_line, *self.signal_lines.values()
            
            if window is not None:
                # Make predictions
                try:
                    predictions = self.predict_signals(window)
                except Exception as e:
                    print(f"Error in predictions: {e}")
                    return self.price_line, *self.signal_lines.values()
                
                # Update signals history
                current_time = processed_data['datetime'].iloc[-1]
                
                for model_name, pred_data in predictions.items():
                    self.signals_history[model_name].append({
                        'time': current_time,
                        'probability': pred_data['probability'],
                        'signal': pred_data['signal']
                    })
                
                # Keep only recent signals (last 100)
                for model_name in self.signals_history:
                    if len(self.signals_history[model_name]) > 100:
                        self.signals_history[model_name] = self.signals_history[model_name][-100:]
                
                # Update price chart
                times = processed_data['datetime'].tail(100)
                prices = processed_data['Close'].tail(100)
                self.price_line.set_data(times, prices)
                
                # Add signal arrows
                self.ax1.clear()
                self.ax1.plot(times, prices, 'white', linewidth=2, label='EUR/USD')
                
                for model_name, color in self.model_colors.items():
                    if predictions[model_name]['signal']:
                        # Add down arrow for sell signal
                        self.ax1.annotate('▼', xy=(current_time, prices.iloc[-1]), 
                                        xytext=(current_time, prices.iloc[-1] + 0.001),
                                        color=color, fontsize=20, ha='center',
                                        annotation_clip=False)
                        print(f"🔻 {model_name} SELL SIGNAL at {current_time}: {predictions[model_name]['probability']:.3f}")
                
                # Update signal strength chart
                for model_name, line in self.signal_lines.items():
                    if self.signals_history[model_name]:
                        signal_times = [s['time'] for s in self.signals_history[model_name]]
                        signal_probs = [s['probability'] for s in self.signals_history[model_name]]
                        line.set_data(signal_times, signal_probs)
                
                # Adjust axes
                if len(times) > 0:
                    self.ax1.set_xlim(times.iloc[0], times.iloc[-1])
                    self.ax1.set_ylim(prices.min() * 0.999, prices.max() * 1.001)
                    
                    self.ax2.set_xlim(times.iloc[0], times.iloc[-1])
                
                # Format x-axis
                self.ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                
                # Redraw legends and labels
                self.ax1.set_title('EUR/USD Live Trading Signals', fontsize=16, color='white')
                self.ax1.set_ylabel('Price', fontsize=12, color='white')
                self.ax1.grid(True, alpha=0.3)
                self.ax1.legend(loc='upper left')
                
                self.ax2.set_ylabel('Signal Strength', fontsize=12, color='white')
                self.ax2.grid(True, alpha=0.3)
                self.ax2.legend(loc='upper right')
                
                # Print current status with prediction results
                prediction_summary = []
                for model_name, pred_data in predictions.items():
                    if pred_data['signal']:
                        prediction_summary.append(f"{model_name}: {pred_data['probability']:.3f}*")
                    else:
                        prediction_summary.append(f"{model_name}: {pred_data['probability']:.3f}")
                
                prediction_str = " | ".join(prediction_summary)
                print(f"⏰ {current_time.strftime('%H:%M:%S')} | Price: {prices.iloc[-1]:.5f} | {prediction_str}")
                
        except Exception as e:
            print(f"Error in update_plot: {e}")
            import traceback
            traceback.print_exc()
        
        return self.price_line, *self.signal_lines.values()
    
    def start_live_prediction(self):
        """Start the live prediction and plotting"""
        print("🚀 Starting live EUR/USD trading signal prediction...")
        print("Models loaded:", list(self.models.keys()))
        print("Signal colors:", self.model_colors)
        print("\nWaiting for sufficient data to start predictions...")
        
        # Start animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=5000, blit=False, cache_frame_data=False)
        
        plt.show()
        
        return ani

# Main execution
if __name__ == "__main__":
    # Create predictor instance
    predictor = LiveTradingPredictor(model_dir=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\sell_models')
    
    # Start live prediction
    animation = predictor.start_live_prediction()