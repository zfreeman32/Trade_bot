# Deep Learning Classification Pipeline for Financial Trading Signals

A production-ready, memory-optimized deep learning pipeline for binary classification of financial trading signals. This system predicts long/short entry signals on EUR/USD 1-minute data using a comprehensive suite of neural architectures and advanced training methodologies.

---

## 🎯 Problem Statement & Approach

### Financial Signal Classification
The pipeline addresses the challenge of predicting binary trading decisions (`long_signal` and `short_signal`) from high-frequency financial time series data. Each timestep requires classification into buy (1) or hold (0) decisions based on technical indicators and engineered features derived from market microstructure.

### Multi-Architecture Ensemble Strategy
Rather than relying on a single model type, the system implements a comprehensive model zoo approach, training and evaluating multiple architectures simultaneously to identify the most robust performers for each signal type.

---

## 🏗️ Architecture & Model Zoo

### Neural Network Architectures
- **Recurrent Models**: LSTM, GRU, Bidirectional LSTM with Attention
- **Convolutional Models**: Conv1D, Conv1D+LSTM hybrid, ResNet-inspired temporal networks
- **Advanced Architectures**: Transformer, Temporal Convolutional Networks (TCN), MultiStream hybrid
- **Traditional ML**: CatBoost, LightGBM, XGBoost, Random Forest

### Modular Design Philosophy
Models are implemented as parameterized builders, enabling dynamic hyperparameter tuning across architectures. Each builder accepts hyperparameter objects and constructs models with variable depth, width, and architectural components.

---

## 🧠 Advanced Training Methodology

### Progressive Training Strategy
**Phase 1 - Warmup**: Models train on 20% of data to establish stable weight initialization and prevent early convergence to poor local minima.

**Phase 2 - Fine-tuning**: Full dataset training with reduced learning rates, leveraging the stable foundation from Phase 1 to achieve better generalization.

### Hyperparameter Optimization
- **Bayesian Optimization**: Uses Keras Tuner's Bayesian search for efficient exploration of hyperparameter space
- **Comprehensive Parameter Coverage**: Every layer parameter is tunable including activation functions, initializers, regularization strategies, constraints, dropout rates, and architecture-specific parameters
- **Stability-First Design**: Built-in regularization (L1/L2), constraints (max_norm, unit_norm), and multiple initialization strategies for robust training
- **Early Termination**: Implements baseline performance thresholds and NaN detection to skip underperforming configurations
- **Resource Management**: Limits trial counts and consecutive failures to prevent resource exhaustion

### Time Series Cross-Validation
- **Temporal Integrity**: Uses TimeSeriesSplit to respect chronological order and prevent data leakage
- **Robustness Testing**: Evaluates top hyperparameter sets across multiple folds to identify consistently performing models
- **Stability Filtering**: Excludes hyperparameter sets with excessive NaN validation results

---

## 📊 Feature Engineering & Data Pipeline

### Advanced Feature Transformation
**Volume Processing**: Applies log transformation, winsorization (1st/99th percentiles), and percentile ranking to handle volume's extreme skewness and outliers.

**Temporal Lag Features**: Creates lagged versions of target signals and key technical indicators (RSI, CCI, EFI, CMO, ROC, ROCR) using optimized lag periods derived from prior analysis.

**Rolling Statistics**: Computes rolling means and standard deviations across lagged target features to capture medium-term signal persistence patterns.

### Sliding Window Architecture
Converts tabular financial data into supervised learning format using sliding windows:
- **Input**: [timesteps × features] arrays capturing historical context
- **Target**: Binary signal at current timestep
- **Memory Optimization**: Chunked processing prevents memory overflow on large datasets

### Robust Preprocessing Pipeline
- **NaN Handling**: Multi-stage NaN detection and replacement using column means
- **Scaling**: RobustScaler with 10th/90th percentile bounds to handle outliers
- **Stability Checks**: Constant feature detection and noise injection
- **Value Clipping**: Prevents extreme values that cause training instability

---

## ⚖️ Class Imbalance & Loss Functions

### Dynamic Class Weighting
Computes balanced class weights from training distributions to address natural imbalance in trading signals where hold positions typically dominate.

### Focal Loss Implementation
Implements focal loss with tunable gamma (focusing parameter) and alpha (class weighting) to:
- **Down-weight Easy Examples**: Reduces loss contribution from confident predictions
- **Focus on Hard Examples**: Increases gradient signal from misclassified samples
- **Handle Imbalance**: Provides class-specific weighting beyond simple rebalancing

---

## 🚀 Memory Optimization & Scalability

### Production-Scale Memory Management
**Chunked Processing**: All major operations (windowing, scaling, training) process data in configurable chunks to handle datasets exceeding available RAM.

**Streaming Datasets**: Uses tf.data pipelines with prefetching and batching to minimize memory footprint during training.

**Limited Step Training**: Caps steps per epoch to reduce training time while maintaining convergence, enabling faster hyperparameter exploration.

### Resource Configuration
- **GPU Auto-Detection**: Automatically configures GPU memory growth and mixed precision training
- **CPU Optimization**: Falls back to optimized CPU threading when GPUs unavailable
- **Memory Monitoring**: Tracks RAM and GPU memory usage throughout pipeline execution

---

## 📈 Training Stability & Robustness

### Comprehensive Callback System
- **Learning Rate Scheduling**: Cosine annealing with warmup for stable convergence
- **Early Stopping**: Multiple criteria including validation loss, accuracy thresholds, and NaN detection
- **Gradient Clipping**: Prevents exploding gradients through value-based clipping
- **Memory Monitoring**: Real-time GPU memory tracking

### Error Recovery & Fault Tolerance
- **Retry Logic**: Multiple attempts for failed training runs with session clearing
- **NaN Safety**: Immediate termination and retry when NaN losses detected
- **Model Validation**: Comprehensive testing of model builders before training begins

---

## 🔄 Dual Signal Processing

### Separate Pipelines for Long/Short Signals
The system maintains independent training pipelines for buy (`long_signal`) and sell (`short_signal`) decisions:

**Buy Pipeline** (`buy_classification_model_train.py`):
- Optimized lag periods: [61, 93, 64, 60, 77]
- Focus on long entry signal prediction
- Results saved to buy-specific models directory

**Sell Pipeline** (`sell_classification_model_train.py`):
- Optimized lag periods: [70, 24, 10, 74, 39]  
- Focus on short entry signal prediction
- Results saved to sell-specific models directory

This separation allows for signal-specific feature engineering and hyperparameter optimization, acknowledging that long and short market dynamics may require different modeling approaches.

---

## 📋 Configuration Management

### Centralized Configuration System
The `ClassificationConfigManager` provides centralized control over:
- **Data Parameters**: File paths, target columns, window sizes
- **Memory Settings**: Sample limits, chunk sizes, streaming configurations
- **Training Parameters**: Batch sizes, epochs, learning rates
- **Model Selection**: Architecture subsets for training
- **Hyperparameter Bounds**: Trial limits and failure thresholds

### Environment Adaptability
Configurations automatically adapt to available computational resources, scaling memory usage and training intensity based on detected hardware capabilities.

---

## 🎯 Evaluation & Model Selection

### Comprehensive Metrics Suite
- **Standard Classification**: Accuracy, Precision, Recall, AUC
- **Financial-Specific**: Directional accuracy for trading signal evaluation
- **Stability Metrics**: Cross-validation variance and convergence behavior

### Model Persistence & Deployment
Trained models are saved in architecture-specific directories with comprehensive hyperparameter logging, enabling easy model comparison and production deployment.

## Classification Models
- build_LSTM_classifier,
- build_GRU_classifier,
- build_Conv1D_classifier,
- build_Conv1D_LSTM_classifier,
- build_BiLSTM_Attention_classifier,
- build_Transformer_classifier
- build_BiLSTM_Attention_classifier
- build_MultiStream_Hybrid_classifier
- build_ResNet_classifier
- build_TCN_classifier
