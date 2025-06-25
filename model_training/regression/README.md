# Multi-Step Time Series Regression Pipeline for Financial Forecasting

This project implements a robust, extensible deep learning pipeline for multi-step regression forecasting, tailored specifically for high-frequency financial data (e.g., EUR/USD minute bars). It incorporates time series preprocessing, feature engineering, model training, evaluation, and experiment logging.

## 🚀 Project Overview

The core functionality includes:

- Custom data preprocessing (log transform, lag features, volume engineering)
- Sliding window time series framing
- Support for multiple deep learning architectures (LSTM, GRU, CNN, TCN, Transformer, etc.)
- Automated hyperparameter tuning via Keras Tuner
- K-Fold cross-validation for robust model selection
- Progressive training strategy to stabilize learning
- Evaluation metrics for multi-step forecasting
- Visualization and export of results

## 🎯 Target Variables

### Returns Prediction Models
- **`returns_1`**: 1-period ahead return forecasting for short-term momentum prediction
- **`returns_5`**: 5-period ahead return forecasting for medium-term trend analysis

Both models utilize the same comprehensive architecture suite but are optimized independently for their respective forecasting horizons, enabling precise magnitude and timing predictions for trading applications.

---

## ⚙️ Key Components

### 1. **Configuration Management** (`ConfigManager`)
Centralized settings including file paths, feature engineering, model selections, and training hyperparameters. Supports JSON save/load.

### 2. **Data Preprocessing**
- Cleans raw price and volume features
- Applies log and winsorized transforms to address skew/outliers
- Adds lag-based features and technical indicators
- Filters based on a curated list of important features

> Techniques: log1p transform, winsorization, rolling mean/std, lag features

### 3. **Sliding Window Creation**
Converts time series into supervised learning format:
- Input: [lookback_window x features]
- Target: [forecast_horizon steps of target variable]

> Utilizes `numpy.sliding_window_view` for efficient sequence framing

### 4. **Scaling**
Uses `RobustScaler` to normalize features and targets. Handles NaNs, infs, and clips outliers for training stability.

### 5. **Model Architectures**
Modular builders support:
- RNNs: LSTM, GRU, SimpleRNN
- CNNs: Conv1D, Conv1DPooling
- Hybrid: Conv1D-LSTM, LSTM-CNN, ResNet, MultiStream
- Sequence Models: Transformer, TCN

### 6. **Losses & Metrics**
- `asymmetric_loss`: penalizes under-forecasting more than over-forecasting
- `directional_accuracy_metric`: measures trend prediction correctness

> Domain-specific: designed for financial applications where prediction *direction* is critical

### 7. **Hyperparameter Tuning**
Uses Keras Tuner (Hyperband) to explore architecture and learning rate variations. Top N models are passed to cross-validation.

### 8. **Cross-Validation**
- 5-Fold CV on best hyperparameter sets
- Calculates mean/variance for val_loss, val_mae, val_mse
- Robustly selects generalizable configurations

> Concept: model selection based on generalization, not overfit to tuning split

### 9. **Progressive Training Strategy**
Two-stage training:
1. Train on 20% subset for fast convergence
2. Fine-tune on full set with reduced learning rate

> Mitigates overfitting and helps early stabilization of training

### 10. **Evaluation & Visualization**
- Metrics: RMSE, MAE, R² (step-wise + overall)
- Plots: loss curves, forecast horizon errors, example prediction curves
- Saves all models and logs results to file

---

## 📈 Training Pipeline

### Returns Model Training
```python
# Train returns_1 model (1-period forecasting)
if __name__ == "__main__":
    config = ConfigManager()
    config.target_column = 'returns_1'
    config.forecast_horizon = 1
    train_regression_models(config)

# Train returns_5 model (5-period forecasting)  
if __name__ == "__main__":
    config = ConfigManager()
    config.target_column = 'returns_5'
    config.forecast_horizon = 5
    train_regression_models(config)
```

This initiates the full pipeline for each target:
- Loads config and data
- Preprocesses and frames dataset for specific forecast horizon
- Tunes and validates models
- Trains final model with selected hyperparameters
- Evaluates and logs results
- Saves trained models as `returns_1_regressor.joblib` and `returns_5_regressor.joblib`

---

## 🧠 Key Concepts Used

| Category           | Techniques & Concepts                                      |
|--------------------|------------------------------------------------------------|
| Data Engineering   | Lag features, rolling stats, log/winsor transforms         |
| Deep Learning      | LSTM, CNN, Transformer, ResNet, TCN                        |
| Evaluation         | RMSE, MAE, R², directional accuracy                        |
| Optimization       | Keras Tuner, Hyperband, K-Fold CV                         |
| Training Strategy  | Progressive Training, early stopping, LR scheduling       |
| Software Design    | Modular builders, config manager, logging                 |
| Hardware Support   | GPU auto-detection and memory growth                      |

---

## 🏗️ Available Regression Model Architectures

### Deep Learning Models
- `build_LSTM_model` - Long Short-Term Memory networks
- `build_GRU_model` - Gated Recurrent Units
- `build_SimpleRNN_model` - Basic recurrent networks
- `build_Conv1D_model` - 1D Convolutional networks
- `build_Conv1DPooling_model` - Conv1D with pooling layers
- `build_Conv1D_LSTM_model` - Hybrid CNN-RNN architecture
- `build_LSTM_CNN_Hybrid_model` - Alternative hybrid approach
- `build_Attention_LSTM_model` - LSTM with attention mechanism
- `build_Transformer_model` - Self-attention transformer
- `build_BiLSTM_Attention_model` - Bidirectional LSTM with attention
- `build_ConvLSTM2D_model` - Convolutional LSTM for 2D data
- `build_MultiStream_Hybrid_model` - Multi-path processing
- `build_ResNet_model` - Residual network architecture
- `build_TCN_model` - Temporal Convolutional Networks

### Traditional ML Models
- `build_RandomForestRegressor_model` - Ensemble tree-based regressor
- `build_XGBoostRegressor_model` - Gradient boosting framework
- `build_GradientBoostingRegressor_model` - Scikit-learn gradient boosting
- `build_LightGBMRegressor_model` - Microsoft's gradient boosting
- `build_CatBoostRegressor_model` - Yandex's categorical boosting

## 📊 Model Output Integration

Trained models are saved and can be integrated into trading systems:

```python
# Load trained models for ensemble prediction
import joblib

returns_1_model = joblib.load('models/returns_1_regressor.joblib')
returns_5_model = joblib.load('models/returns_5_regressor.joblib')

# Generate multi-horizon forecasts
short_term_forecast = returns_1_model.predict(live_data)  # 1-period ahead
medium_term_forecast = returns_5_model.predict(live_data)  # 5-period ahead
```

These models provide magnitude and timing predictions that complement binary classification signals in comprehensive trading strategies.