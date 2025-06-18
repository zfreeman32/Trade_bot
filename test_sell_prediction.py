#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
import ta  # Using 'ta' library instead of talib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import os
from tqdm import tqdm
import time

# Configure matplotlib for better visuals
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = 'black'
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['figure.figsize'] = (20, 12)

class TradingModelTester:
    def __init__(self, 
                 sell_model_dir=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\sell_signal_models\sell_models',
                 buy_model_dir=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\long_signal_models\buy_models',
                 lookback_window=120):
        self.sell_model_dir = sell_model_dir
        self.buy_model_dir = buy_model_dir
        self.lookback_window = lookback_window
        self.sell_lag_periods = [70, 24, 10, 74, 39]  # Lag periods for sell signals
        self.buy_lag_periods = [61, 93, 64, 60, 77]   # Lag periods for buy signals
        self.sell_models = {}
        self.buy_models = {}
        
        # Different colors for sell and buy models
        self.model_colors = {
            # Sell models (reds and oranges - bearish colors)
            'LSTM_Sell': 'red',
            'GRU_Sell': 'orange', 
            'Conv1D_Sell': 'darkred',
            'Conv1D_LSTM_Sell': 'orangered',
            # Buy models (greens and blues - bullish colors)
            'LSTM_Buy': 'green',
            'GRU_Buy': 'lime', 
            'Conv1D_Buy': 'darkgreen',
            'Conv1D_LSTM_Buy': 'lightgreen'
        }
        
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.is_scaler_fitted = False
        
        # Define the exact feature list used during training (excluding target and time columns)
        # Based on training script's exclude_columns logic
        self.base_features = [
            # Core technical indicators
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
        
        # Build training features dynamically to match training script
        self.training_features = self.base_features.copy()
        
        # Add lag features to match training script exactly
        # For buy signals, training script uses the same naming as sell signals
        for lag in self.buy_lag_periods:
            self.training_features.append(f'long_signal_lag_{lag}')
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                self.training_features.append(f'{indicator}_lag_{lag}')
        
        # Add rolling stats on lag features (matches training script naming)
        self.training_features.extend(['target_lag_mean', 'target_lag_std'])
        
        # Add volume transformations
        self.training_features.extend(['Volume_log', 'Volume_winsor', 'Volume_rank'])
        
        print(f"Expected {len(self.training_features)} features for model compatibility")
        print("⚠️  Note: Current preprocessing is configured for BUY models only.")
        print("   Sell models may need different lag features if trained separately.")
        
        # Storage for test results
        self.test_results = {
            'timestamps': [],
            'prices': [],
            'sell_predictions': {name.replace('_Sell', ''): [] for name in self.model_colors.keys() if '_Sell' in name},
            'sell_signals': {name.replace('_Sell', ''): [] for name in self.model_colors.keys() if '_Sell' in name},
            'buy_predictions': {name.replace('_Buy', ''): [] for name in self.model_colors.keys() if '_Buy' in name},
            'buy_signals': {name.replace('_Buy', ''): [] for name in self.model_colors.keys() if '_Buy' in name}
        }
        
    def load_models(self):
        """Load all trained models (both sell and buy)"""
        print("Loading trained models...")
        model_files = {
            'LSTM': 'LSTM.h5',
            'GRU': 'GRU.h5', 
            'Conv1D': 'Conv1D.h5',
            'Conv1D_LSTM': 'Conv1D_LSTM.h5'
        }
        
        # Load sell models
        print("Loading sell models...")
        for model_name, filename in model_files.items():
            model_path = os.path.join(self.sell_model_dir, filename)
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(
                        model_path,
                        custom_objects={'focal_loss_fn': self.focal_loss_fn}
                    )
                    
                    # Check model input shape
                    expected_shape = model.input_shape
                    print(f"✅ Loaded {model_name}_Sell model - Expected input shape: {expected_shape}")
                    
                    self.sell_models[model_name] = model
                    
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
                        model_path,
                        custom_objects={'focal_loss_fn': self.focal_loss_fn}
                    )
                    
                    # Check model input shape
                    expected_shape = model.input_shape
                    print(f"✅ Loaded {model_name}_Buy model - Expected input shape: {expected_shape}")
                    
                    self.buy_models[model_name] = model
                    
                except Exception as e:
                    print(f"❌ Error loading {model_name}_Buy: {e}")
            else:
                print(f"❌ Buy model file not found: {model_path}")
        
        print(f"Successfully loaded {len(self.sell_models)} sell models and {len(self.buy_models)} buy models")
        print(f"Sell models: {list(self.sell_models.keys())}")
        print(f"Buy models: {list(self.buy_models.keys())}")
    
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
    
    def fetch_historical_data(self, samples=50000, csv_path=r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled.csv'):
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
                data['Volume'] = np.random.randint(1000, 50000, len(data))
            
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
    
    def generate_sample_data(self, samples=50000):
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
            'Volume': np.random.randint(1000, 50000, len(dates))
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
        """Add lag features exactly as used in training (buy models only)"""
        # Create dummy target column for lag calculation
        data['long_signal'] = 0
        
        # Add buy signal lag features with same naming as training script
        for lag in self.buy_lag_periods:
            data[f'long_signal_lag_{lag}'] = data['long_signal'].shift(lag)
            
            # Add lagged versions of key technical indicators 
            # NOTE: Training script uses same naming for buy signals: {indicator}_lag_{lag}
            for indicator in ['RSI', 'CCI', 'EFI', 'CMO', 'ROC', 'ROCR']:
                if indicator in data.columns:
                    data[f'{indicator}_lag_{lag}'] = data[indicator].shift(lag)
        
        # Add rolling stats exactly as in training script
        lag_cols = [f'long_signal_lag_{lag}' for lag in self.buy_lag_periods]
        data['target_lag_mean'] = data[lag_cols].mean(axis=1)
        data['target_lag_std'] = data[lag_cols].std(axis=1)
        
        # Remove the dummy target column
        data = data.drop(['long_signal'], axis=1)
        
        # Handle missing values from lags
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return data
    
    def preprocess_data(self, data):
        """
        Apply the same preprocessing as used in training
        NOTE: This preprocessing matches the buy model training script.
        Sell models may need different preprocessing if they were trained with different lag features.
        """
        # Add technical indicators
        data = self.calculate_technical_indicators(data)
        
        # Add lag features (currently configured for buy models only)
        data = self.add_lag_features(data)
        
        # Transform volume-based features (matches training script)
        volume_cols = ['Volume']
        for col in volume_cols:
            if col in data.columns:
                data[f'{col}_log'] = np.log1p(data[col])
                q_low, q_high = data[col].quantile(0.01), data[col].quantile(0.99)
                data[f'{col}_winsor'] = data[col].clip(q_low, q_high)
                data[f'{col}_rank'] = data[col].rank(pct=True)
        
        # Select ONLY the features that were used during training
        # This matches the training script's logic of excluding target and time columns
        features = pd.DataFrame()
        missing_features = []
        
        for feature in self.training_features:
            if feature in data.columns:
                features[feature] = data[feature]
            elif feature.startswith('dummy_feature_'):
                # Handle dummy features added for shape matching
                features[feature] = 0
            else:
                # Create missing feature with zeros
                features[feature] = 0
                missing_features.append(feature)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features missing, filled with zeros")
            print(f"Missing features: {missing_features[:10]}...")  # Show first 10
        
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
        
        return features, data
    
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
    
    def scale_features(self, features):
        """Scale features using RobustScaler"""
        if not self.is_scaler_fitted:
            print("Fitting scaler on initial data...")
            self.scaler.fit(features.values)
            self.is_scaler_fitted = True
        
        # Scale the features
        features_scaled = self.scaler.transform(features.values)
        features_scaled = np.clip(features_scaled, -10, 10)  # Cap values
        return features_scaled
    
    def predict_signals(self, window):
        """Make predictions using all loaded models (both sell and buy)"""
        predictions = {
            'sell': {},
            'buy': {}
        }
        
        # Predictions from sell models
        for model_name, model in self.sell_models.items():
            try:
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
                
                # Consider it a signal if probability > 0.5
                is_signal = signal_prob > 0.5
                
                predictions['sell'][model_name] = {
                    'probability': float(signal_prob),
                    'signal': bool(is_signal)
                }
                
            except Exception as e:
                print(f"Error predicting with {model_name}_Sell: {e}")
                predictions['sell'][model_name] = {'probability': 0.0, 'signal': False}
        
        # Predictions from buy models
        for model_name, model in self.buy_models.items():
            try:
                # Make prediction
                pred = model.predict(window, verbose=0)[0]
                
                # Convert to probability/signal strength
                if len(pred) > 1:
                    signal_prob = pred[1]  # Probability of class 1 (long signal)
                else:
                    signal_prob = pred[0]
                
                # Handle potential NaN or inf values
                if np.isnan(signal_prob) or np.isinf(signal_prob):
                    signal_prob = 0.0
                
                # Consider it a signal if probability > 0.5
                is_signal = signal_prob > 0.5
                
                predictions['buy'][model_name] = {
                    'probability': float(signal_prob),
                    'signal': bool(is_signal)
                }
                
            except Exception as e:
                print(f"Error predicting with {model_name}_Buy: {e}")
                predictions['buy'][model_name] = {'probability': 0.0, 'signal': False}
        
        return predictions
    
    def run_backtest(self, historical_data):
        """Run backtest simulation cycling through historical data"""
        print(f"🚀 Starting backtest on {len(historical_data)} historical samples...")
        
        # Preprocess all data first
        print("Preprocessing historical data...")
        features, processed_data = self.preprocess_data(historical_data)
        
        # Scale features
        print("Scaling features...")
        features_scaled = self.scale_features(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        
        # Calculate minimum samples needed (using buy lag periods for buy models)
        # Note: Sell models would need their own lag periods if trained separately
        min_samples_needed = self.lookback_window + max(self.buy_lag_periods)
        
        print(f"Starting predictions from sample {min_samples_needed} onwards...")
        
        # Run predictions for each valid time point
        valid_samples = len(features_scaled_df) - self.lookback_window
        
        with tqdm(total=valid_samples, desc="Processing samples") as pbar:
            for i in range(min_samples_needed, len(features_scaled_df)):
                # Create sliding window
                window = self.create_sliding_windows(features_scaled_df, i - self.lookback_window)
                
                if window is not None:
                    # Make predictions
                    predictions = self.predict_signals(window)
                    
                    # Store results
                    self.test_results['timestamps'].append(processed_data['datetime'].iloc[i])
                    self.test_results['prices'].append(processed_data['Close'].iloc[i])
                    
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
                
                # Print progress every 1000 samples
                if i % 1000 == 0:
                    current_time = processed_data['datetime'].iloc[i]
                    current_price = processed_data['Close'].iloc[i]
                    
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
                    
                    print(f"Sample {i}/{len(features_scaled_df)} | {current_time.strftime('%Y-%m-%d %H:%M')} | Price: {current_price:.5f} | {' | '.join(signal_summary)}")
        
        print("✅ Backtest completed!")
        return self.test_results
    
    def plot_results(self):
        """Plot the backtest results"""
        if not self.test_results['timestamps']:
            print("No results to plot!")
            return
        
        print("📊 Plotting backtest results...")
        
        # Create the plot
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 18), 
                                              gridspec_kw={'height_ratios': [3, 2, 2, 1]})
        
        times = pd.to_datetime(self.test_results['timestamps'])
        prices = self.test_results['prices']
        
        # Plot 1: Price chart with signals
        ax1.plot(times, prices, 'white', linewidth=1, label='EUR/USD', alpha=0.8)
        
        # Add sell signal markers (down arrows)
        for model_name in self.test_results['sell_signals'].keys():
            signal_times = []
            signal_prices = []
            
            for i, is_signal in enumerate(self.test_results['sell_signals'][model_name]):
                if is_signal:
                    signal_times.append(self.test_results['timestamps'][i])
                    signal_prices.append(self.test_results['prices'][i])
            
            if signal_times:
                color = self.model_colors[f'{model_name}_Sell']
                ax1.scatter(signal_times, signal_prices, color=color, marker='v', 
                          s=50, alpha=0.7, label=f'{model_name} Sell ↓')
        
        # Add buy signal markers (up arrows)
        for model_name in self.test_results['buy_signals'].keys():
            signal_times = []
            signal_prices = []
            
            for i, is_signal in enumerate(self.test_results['buy_signals'][model_name]):
                if is_signal:
                    signal_times.append(self.test_results['timestamps'][i])
                    signal_prices.append(self.test_results['prices'][i])
            
            if signal_times:
                color = self.model_colors[f'{model_name}_Buy']
                ax1.scatter(signal_times, signal_prices, color=color, marker='^', 
                          s=50, alpha=0.7, label=f'{model_name} Buy ↑')
        
        ax1.set_title('EUR/USD Price with Trading Signals (↓ Sell, ↑ Buy)', fontsize=16, color='white')
        ax1.set_ylabel('Price', fontsize=12, color='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', ncol=2)
        
        # Plot 2: Sell Signal probabilities
        for model_name in self.test_results['sell_predictions'].keys():
            probs = self.test_results['sell_predictions'][model_name]
            color = self.model_colors[f'{model_name}_Sell']
            ax2.plot(times, probs, color=color, linewidth=1.5, label=f'{model_name} Sell', alpha=0.8)
        
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Signal Threshold')
        ax2.set_title('Sell Model Prediction Probabilities', fontsize=14, color='white')
        ax2.set_ylabel('Sell Signal Probability', fontsize=12, color='white')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        
        # Plot 3: Buy Signal probabilities
        for model_name in self.test_results['buy_predictions'].keys():
            probs = self.test_results['buy_predictions'][model_name]
            color = self.model_colors[f'{model_name}_Buy']
            ax3.plot(times, probs, color=color, linewidth=1.5, label=f'{model_name} Buy', alpha=0.8)
        
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Signal Threshold')
        ax3.set_title('Buy Model Prediction Probabilities', fontsize=14, color='white')
        ax3.set_ylabel('Buy Signal Probability', fontsize=12, color='white')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper right')
        
        # Plot 4: Signal count summary
        signal_counts = {}
        
        # Count sell signals
        for model_name in self.test_results['sell_signals'].keys():
            signal_count = sum(self.test_results['sell_signals'][model_name])
            signal_counts[f'{model_name}_Sell'] = signal_count
        
        # Count buy signals
        for model_name in self.test_results['buy_signals'].keys():
            signal_count = sum(self.test_results['buy_signals'][model_name])
            signal_counts[f'{model_name}_Buy'] = signal_count
        
        models = list(signal_counts.keys())
        counts = list(signal_counts.values())
        colors = [self.model_colors[model] for model in models]
        
        bars = ax4.bar(models, counts, color=colors, alpha=0.7)
        ax4.set_title('Total Signal Count by Model', fontsize=14, color='white')
        ax4.set_ylabel('Number of Signals', fontsize=12, color='white')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', color='white', fontweight='bold')
        
        # Format time axes
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\n📈 Backtest Summary:")
        print(f"Total samples processed: {len(self.test_results['timestamps'])}")
        print(f"Time period: {times.min().strftime('%Y-%m-%d %H:%M')} to {times.max().strftime('%Y-%m-%d %H:%M')}")
        print(f"Price range: {min(prices):.5f} - {max(prices):.5f}")
        
        print("\n🎯 Sell Signal Statistics:")
        total_samples = len(self.test_results['timestamps'])
        for model_name in self.test_results['sell_predictions'].keys():
            signal_count = sum(self.test_results['sell_signals'][model_name])
            signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
            avg_prob = np.mean(self.test_results['sell_predictions'][model_name])
            max_prob = np.max(self.test_results['sell_predictions'][model_name])
            
            print(f"{model_name:12}: {signal_count:4d} signals ({signal_rate:5.2f}%) | "
                  f"Avg prob: {avg_prob:.3f} | Max prob: {max_prob:.3f}")
        
        print("\n🎯 Buy Signal Statistics:")
        for model_name in self.test_results['buy_predictions'].keys():
            signal_count = sum(self.test_results['buy_signals'][model_name])
            signal_rate = (signal_count / total_samples) * 100 if total_samples > 0 else 0
            avg_prob = np.mean(self.test_results['buy_predictions'][model_name])
            max_prob = np.max(self.test_results['buy_predictions'][model_name])
            
            print(f"{model_name:12}: {signal_count:4d} signals ({signal_rate:5.2f}%) | "
                  f"Avg prob: {avg_prob:.3f} | Max prob: {max_prob:.3f}")

def main():
    """Main function to run the trading model test"""
    # Configuration
    SELL_MODEL_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\sell_signal_models\sell_models'
    BUY_MODEL_DIR = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\models\long_signal_models\buy_models'
    SAMPLES = 50000  # Number of historical samples to test
    
    # Initialize tester
    print("🔧 Initializing Trading Model Tester...")
    tester = TradingModelTester(sell_model_dir=SELL_MODEL_DIR, buy_model_dir=BUY_MODEL_DIR)
    
    # Load models
    tester.load_models()
    
    if not tester.sell_models and not tester.buy_models:
        print("❌ No models loaded! Please check the model directory paths.")
        return
    
    # Fetch historical data
    historical_data = tester.fetch_historical_data(samples=SAMPLES)
    
    # Run backtest
    results = tester.run_backtest(historical_data)
    
    # Plot results
    tester.plot_results()
    
    print("\n✅ Test completed successfully!")
    print(f"Loaded {len(tester.sell_models)} sell models and {len(tester.buy_models)} buy models")

if __name__ == "__main__":
    main()

#%%