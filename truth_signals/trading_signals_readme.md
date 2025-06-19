# EURUSD Trading Signals - Technical Overview

A high-performance trading signal generation and backtesting system for EURUSD forex data using dynamic programming optimization and GPU acceleration.

## 🎯 Approach

### Signal Generation (`find_truth_signals.py`)

**Core Algorithm**: Dynamic Programming Optimization
- Uses backward induction to find mathematically optimal, non-overlapping trade sequences
- Maximizes total profit across the entire dataset rather than greedy local optimization
- Ensures trades don't overlap in time, creating realistic trading scenarios

**Technical Implementation**:
- **GPU Processing**: CUDA kernels with CuPy for parallel trade evaluation across thousands of potential entry/exit combinations
- **CPU Fallback**: Numba JIT compilation with parallel processing (`prange`) for systems without CUDA
- **Memory Efficiency**: Processes large datasets by transferring data to GPU memory and using optimized array operations

**Filtering Criteria**:
- **ATR-based profit thresholds**: Trades must exceed volatility-adjusted minimum profits
- **Volume confirmation**: Only considers entries when volume exceeds moving average (reduces false signals)
- **Minimum absolute profit**: Hard threshold to filter out noise trades

**Why This Approach**:
- Dynamic programming guarantees globally optimal solutions within constraints
- GPU acceleration makes exhaustive search computationally feasible for large datasets
- Volume and ATR filters align with actual market microstructure principles

### Technical Indicators

**Average True Range (ATR)**:
- Custom JIT-compiled implementation for speed
- Used for volatility-adjusted profit thresholds
- Ensures trades are meaningful relative to market volatility

**Volume Moving Average**:
- Simple moving average to identify above-average activity periods
- Filters out low-conviction signals during quiet market periods

### Backtesting (`backtest_truth_signals.py`)

**Dual Implementation Strategy**:

1. **Professional Framework** (`Backtesting.py`):
   - Industry-standard library with realistic execution simulation
   - Handles position sizing, slippage, and commission automatically
   - Provides comprehensive performance metrics and equity curves

2. **Manual Vectorized Implementation**:
   - Custom loop-based execution for full control and verification
   - Immediate signal execution without library overhead
   - Direct profit calculation for transparency and debugging

**Why Both Approaches**:
- Cross-validation of results between different execution models
- Professional library provides realistic market simulation
- Manual implementation offers complete transparency and customization

## 🛠 Technology Stack

### Performance Optimization
- **CuPy**: GPU-accelerated NumPy replacement for CUDA operations
- **Numba**: JIT compilation for CPU-intensive loops and mathematical operations
- **CUDA Kernels**: Custom parallel processing for trade evaluation
- **Vectorized Operations**: NumPy broadcasting for efficient array computations

### Core Libraries
- **Pandas**: Data manipulation and time series handling
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Visualization of results and trade analysis

### Hardware Detection
- **Automatic GPU/CPU Detection**: Runtime optimization based on available hardware
- **Multi-core Utilization**: Parallel processing across available CPU cores
- **Memory Management**: Efficient handling of large datasets in GPU/CPU memory

## 🧠 Key Design Decisions

**Dynamic Programming Choice**:
- Chosen over machine learning approaches for deterministic, mathematically optimal results
- Avoids overfitting issues common in ML-based trading systems
- Provides explainable signals based on clear profit criteria

**GPU Acceleration Priority**:
- CUDA implementation for O(n²) trade evaluation complexity
- Enables processing of large forex datasets (100K+ bars) in reasonable time
- Graceful degradation to optimized CPU processing when GPU unavailable

**Dual Backtesting Validation**:
- Professional library ensures realistic execution assumptions
- Manual implementation provides algorithmic transparency
- Comparison between approaches validates signal quality

**Signal Format Design**:
- Binary signals (0/1) for easy integration with trading systems
- Separate long/short/close columns for clear position management
- Non-overlapping trades ensure realistic capital allocation

## 📊 Output Signals

The system generates three binary signal types:
- `long_signal`: Entry points for long positions
- `short_signal`: Entry points for short positions  
- `close_position`: Exit points for current positions

These signals represent mathematically optimal entry/exit points based on the dynamic programming solution, filtered by volume and volatility criteria to ensure practical trading viability.