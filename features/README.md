# EURUSD Feature Engineering & Analysis Pipeline

A production-grade data engineering pipeline for systematic feature generation, analysis, and selection in high-frequency forex trading data. The pipeline implements multi-stage data transformations, advanced feature selection algorithms, and optimized data processing architectures.

## 🏗️ Data Engineering Architecture

### Pipeline Data Flow

```
Raw EURUSD OHLCV Data
         ↓
┌─────────────────────────────────────┐
│     Stage 1: Technical Indicators   │
│   • 147 TA-Lib transformations     │
│   • Vectorized calculations        │
│   • Memory-efficient processing    │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Stage 2: Strategy Features      │
│   • Parallel strategy execution    │
│   • Dynamic feature synthesis      │
│   • Result aggregation & merging   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Stage 3: Feature Filtering      │
│   • Constant feature removal       │
│   • Collinearity analysis          │
│   • Stability filtering           │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Stage 4: Advanced Selection     │
│   • Multi-method ensemble          │
│   • Cross-validation scoring       │
│   • Importance ranking             │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│     Stage 5: Dataset Creation       │
│   • Target-specific datasets       │
│   • Optimized storage format       │
└─────────────────────────────────────┘
```

## 📊 Technical Implementation

### Stage 1: Technical Indicator Generation

**Data Transformation Engine (`all_indicators.py`)**

The indicator generation stage implements a comprehensive technical analysis transformation layer using TA-Lib as the core computation engine.

**Feature Categories & Engineering:**

```python
# Overlap Studies (21 indicators)
df['UPPERBAND'], df['MIDDLEBAND'], df['LOWERBAND'] = talib.BBANDS(close, timeperiod=20)
df['EMA_10'] = talib.EMA(close, timeperiod=10)
df['KAMA'] = talib.KAMA(close, timeperiod=30)

# Momentum Indicators (30 indicators)
df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACD(close)
df['RSI'] = talib.RSI(close, timeperiod=14)

# Volume Indicators (3 indicators)
df['AD'] = talib.AD(high, low, close, volume)
df['OBV'] = talib.OBV(close, volume)

# Volatility Indicators (3 indicators)
df['ATR'] = talib.ATR(high, low, close, timeperiod=14)

# Price Transform (4 indicators)
df['AVGPRICE'] = talib.AVGPRICE(open_price, high, low, close)

# Cycle Indicators (5 indicators)
df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)

# Candlestick Patterns (61 indicators)
df['CDL2CROWS'] = talib.CDL2CROWS(open_price, high, low, close)
```

**Custom Feature Engineering:**
```python
# Statistical Features
df['rolling_mean'] = df['Close'].rolling(window=14).mean()
df['rolling_std'] = df['Close'].rolling(window=14).std()
df['z_score'] = (df['Close'] - df['rolling_mean']) / df['rolling_std']

# Advanced Technical Indicators
df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
df['HMA'] = talib.WMA(2 * talib.WMA(close, timeperiod=10 // 2) - talib.WMA(close, timeperiod=10), timeperiod=int(np.sqrt(10)))
```

**Engineering Decisions:**
- **Memory Efficiency**: Direct DataFrame column assignment to minimize memory overhead
- **Vectorized Operations**: Leverage NumPy/TA-Lib vectorization for performance
- **Error Handling**: Comprehensive exception handling for edge cases in financial data
- **Data Type Optimization**: Automatic type conversion to float64 for numerical stability

### Stage 2: Strategy Feature Generation

**Domain-Specific Trading Strategy Engine**

Implements a comprehensive strategy library that generates advanced trading signals by combining multiple technical indicators into sophisticated trading logic. Each strategy encapsulates proven trading methodologies and market patterns.

**Strategy Categories & Implementation:**

**Breakout Strategies:**
```python
# ADX Breakout Strategy
def adx_breakouts_signals(stock_df, highest_length=15, adx_length=14, adx_level=40):
    """
    Combines trend strength (ADX) with price breakouts
    - Rolling highest high detection
    - ADX trend strength filtering
    - Breakout confirmation logic
    """
    highest = stock_df['High'].rolling(window=highest_length).max()
    adx = trend.ADXIndicator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=adx_length).adx()
    breakout_condition = (adx > adx_level) & (stock_df['Close'] > (highest + offset))

# ATR + SMA Breakout Strategy
def atr_high_sma_breakouts_le(df, atr_period=14, sma_period=100):
    """
    Ken Calhoun's ATR-based breakout system
    - ATR volatility filtering
    - SMA trend confirmation
    - Wide-range candle validation
    - Volume surge confirmation
    """
```

**Mean Reversion Strategies:**
```python
# Bollinger Bands Mean Reversion
def bollinger_bands_le_signals(data, length=20, num_devs_dn=2.0):
    """
    Price oversold condition detection
    - Lower band penetration signals
    - Statistical mean reversion logic
    """

# Accumulation/Distribution Strategy
def acc_dist_strat(df, length=4, factor=0.75, vol_ratio=1):
    """
    Volume-price relationship analysis
    - Range contraction detection
    - Volume surge confirmation
    - Support level breakout validation
    """
```

**Trend Following Strategies:**
```python
# ADX Trend Strategy
def ADXTrend_signals(stock_df, length=14, trend_level=25, max_level=50):
    """
    Multi-condition trend identification
    - ADX momentum analysis
    - Moving average trend confirmation
    - Dynamic threshold adjustment
    """

# ATR Trailing Stop Systems
def atr_trailing_stop_le_signals(stock_df, atr_period=14, atr_factor=3):
    """
    Volatility-adjusted trend following
    - ATR-based stop loss calculation
    - Dynamic trailing stop mechanism
    """
```

**Pattern Recognition Strategies:**
```python
# Consecutive Bars Pattern
def cons_bars_up_le_signals(stock_df, consec_bars_up=4):
    """
    Momentum continuation pattern detection
    - Rolling sum pattern analysis
    - Transition point identification
    """

# Camarilla Pivot Points
def camarilla_strategy(stock_df):
    """
    Intraday support/resistance levels
    - Mathematical pivot calculations
    - Multi-level breakout detection
    """
```

**Parallel Processing Architecture (`create_features_dataset.py`)**

```python
def apply_strategies_parallel(strategies, stock_df, max_workers=None):
    """
    ProcessPoolExecutor-based strategy application
    - CPU-bound task parallelization
    - Memory-isolated worker processes
    - Error resilience with worker failure handling
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, (name, func)) for name, func in strategies.items()]
```

**Strategy Engineering Principles:**
- **Signal Standardization**: All strategies return standardized 0/1 signals
- **Multi-Timeframe Analysis**: Strategies incorporate various lookback periods
- **Volatility Adaptation**: ATR-based dynamic parameter adjustment
- **Volume Integration**: Volume confirmation in breakout strategies
- **Risk Management**: Built-in stop-loss and position sizing logic

**Data Processing Optimizations:**
```python
@timer
def load_data(file_path):
    """Memory-optimized data loading with type inference control"""
    stock_df = pd.read_csv(file_path, dtype={'Date': str, 'Time': str}, low_memory=False)
    
    # Optimized datetime conversion
    stock_df['Datetime'] = pd.to_datetime(stock_df['Date'] + ' ' + stock_df['Time'], 
                                         format='%Y%m%d %H:%M:%S', errors='coerce')
    
    # Memory usage optimization
    for col in stock_df.select_dtypes(include=['float']):
        stock_df[col] = pd.to_numeric(stock_df[col], downcast='float')
```

**Result Aggregation Strategy:**
```python
def merge_results(results_dict, base_df):
    """
    Intelligent feature merging with conflict resolution
    - Index-aligned merging for time series consistency
    - Memory-efficient column initialization
    - Duplicate feature handling
    """
    signals_df = base_df.copy()
    
    # Pre-allocate columns to avoid reindexing overhead
    all_new_cols = set()
    for result_df in results_dict.values():
        if result_df is not None:
            all_new_cols.update([col for col in result_df.columns if col not in signals_df.columns])
    
    # Initialize with NaN for memory efficiency
    for col in all_new_cols:
        signals_df[col] = np.nan
```

### Stage 3: Feature Filtering Pipeline

**Multi-Stage Filtering Architecture**

Implements a cascading filter system to remove problematic features before expensive selection algorithms.

**Stage 3.1: Constant Feature Detection**
```python
def _remove_constant_features(self, numeric_features, threshold=1e-10):
    """Remove features with near-zero variance using vectorized operations"""
    constant_features = []
    for col in numeric_features:
        if self.df[col].std() <= threshold:
            constant_features.append(col)
    return constant_features
```

**Stage 3.2: Collinearity Analysis**
```python
def calculate_feature_correlation_matrix(df, numeric_features, correlation_threshold=0.95):
    """
    Memory-efficient correlation calculation using chunked processing
    - Handles large feature sets (1000+ features)
    - Identifies highly correlated pairs
    - Variance-based feature selection for correlated pairs
    """
    chunk_size = 100
    correlation_matrix = np.eye(len(numeric_features))
    
    # Block-wise correlation calculation
    for i in range(0, len(numeric_features), chunk_size):
        for j in range(i, len(numeric_features), chunk_size):
            chunk_corr = np.corrcoef(chunk_data_i.T, chunk_data_j.T)
            correlation_matrix[i:end_i, j:end_j] = chunk_corr
```

**Stage 3.3: Stability Filtering**
```python
def quick_stability_check(df, numeric_features, target_col, n_splits=3, cv_threshold=1.0):
    """
    Time series stability assessment using coefficient of variation
    - TimeSeriesSplit for temporal validation
    - Random Forest importance variance analysis
    - Early elimination of unstable features
    """
    for train_idx, test_idx in tscv.split(X):
        model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        for feature, importance in zip(numeric_features, model.feature_importances_):
            importance_over_time[feature].append(importance)
```

### Stage 4: Advanced Feature Selection

**Multi-Method Ensemble Selection Architecture**

Implements five distinct feature selection methodologies with ensemble voting and cross-validation scoring.

**Method Implementation Matrix:**

| Method | Algorithm | Scoring | Validation |
|--------|-----------|---------|------------|
| Statistical | Mutual Information + Percentile | MI Score | Cross-validation |
| RFE | Recursive Feature Elimination | Feature Ranking | TimeSeriesSplit |
| Regularization | L1/Lasso + LogisticCV | Coefficient Magnitude | Built-in CV |
| Model-Based | Random Forest Importance | Tree Importance | Cross-validation |
| Ensemble | Weighted Voting | Combined Score | All Methods |

**Ensemble Scoring Algorithm:**
```python
def method_5_ensemble_ranking(self, X, y):
    """
    Multi-method ensemble with normalized scoring
    - Vote aggregation across selection methods
    - Score normalization for method combination
    - Weighted feature ranking (60% votes, 40% scores)
    """
    ensemble_scores = {}
    for feat in all_features:
        vote_score = feature_votes[feat] / len(methods)
        avg_score = feature_scores_sum[feat] / max(1, feature_votes[feat])
        ensemble_scores[feat] = 0.6 * vote_score + 0.4 * avg_score
```

**SHAP Analysis Engine:**
```python
def analyze_shap_values(self, sample_size=100000, max_time_minutes=30, batch_size=500):
    """
    Memory-efficient SHAP value calculation with adaptive sampling
    - Batch processing for memory management
    - Timeout protection for large datasets
    - Partial result preservation during computation
    """
    # Adaptive memory management
    estimated_memory_per_row = X.memory_usage().sum() / len(X) / 1024 / 1024
    estimated_total_needed = estimated_memory_per_row * sample_size * 10
    
    if estimated_total_needed > available_memory * 0.5:
        adjusted_sample_size = int(available_memory * 0.5 / (estimated_memory_per_row * 10))
        sample_size = adjusted_sample_size
```

**Time Series Stability Analysis:**
```python
def analyze_time_series_stability(self, n_splits=5, selected_features=None):
    """
    Feature importance stability across time periods
    - TimeSeriesSplit for temporal cross-validation
    - Coefficient of variation for stability metrics
    - Trend analysis for feature degradation detection
    """
    stability_metrics[feature] = {
        'mean_importance': np.mean(importances),
        'coefficient_of_variation': np.std(importances) / np.mean(importances),
        'trend': np.polyfit(range(len(importances)), importances, 1)[0]
    }
```

### Stage 5: Target-Specific Dataset Creation

**Optimized Dataset Construction (`create_important_features_dataset.py`)**

**Chunked Processing Architecture:**
```python
# Memory-efficient chunked processing
chunks = pd.read_csv(input_csv, chunksize=100000)
filtered_chunks = [chunk[available_columns] for chunk in chunks]
pd.concat(filtered_chunks).to_csv(output_csv, index=False)
```

**Feature Priority Management:**
```python
def dedupe_preserve_order(seq):
    """Deduplication while maintaining feature selection order"""
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

# Essential feature preservation
essential_columns = [
    'Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'datetime', 'long_signal', 'short_signal', 'close_position'
]
```

## 🔧 Data Engineering Optimizations

### Memory Management
- **Type Optimization**: Automatic downcast to reduce memory footprint
- **Chunked Processing**: Handle datasets exceeding available RAM
- **Lazy Evaluation**: On-demand feature calculation to minimize memory usage
- **Garbage Collection**: Strategic memory cleanup in long-running processes

### Computational Efficiency
- **Vectorized Operations**: NumPy/Pandas vectorization for mathematical operations
- **Parallel Processing**: Multi-core utilization for independent computations
- **Caching**: Intermediate result caching for expensive operations
- **Batch Processing**: Optimized batch sizes for memory/speed trade-offs

### Data Integrity
- **Missing Value Handling**: Comprehensive NaN and infinite value treatment
- **Type Safety**: Consistent data type enforcement across pipeline stages
- **Index Alignment**: Time series index consistency throughout transformations
- **Validation**: Data quality checks at each pipeline stage

### Scalability Architecture
- **Modular Design**: Independent stage execution for horizontal scaling
- **Error Isolation**: Worker process isolation prevents cascade failures
- **Progressive Results**: Intermediate result saving for long-running processes
- **Resource Monitoring**: Real-time memory and CPU usage tracking

## 📈 Feature Engineering Methodology

### Target Engineering Strategy
- **Profit-Aware Targets**: Forward-looking profit calculation with stop-loss integration
- **Directional Accuracy**: Price movement direction prediction enhancement
- **Risk-Adjusted Signals**: Signal quality assessment based on historical performance

### Feature Selection Philosophy
- **Multi-Method Validation**: Cross-validation across different selection algorithms
- **Temporal Stability**: Feature importance consistency across time periods
- **Ensemble Voting**: Democratic feature selection reducing method bias
- **Performance-Based Filtering**: Actual trading performance consideration in selection

### Data Processing Principles
- **Time Series Integrity**: Preservation of temporal relationships in all transformations
- **Look-Ahead Bias Prevention**: Strict future data isolation in feature engineering
- **Market Regime Awareness**: Feature stability across different market conditions
- **High-Frequency Optimization**: Specialized handling for 1-minute resolution data