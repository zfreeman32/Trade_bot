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

```python
if __name__ == "__main__":
    train_regression_models(config)
```
This initiates the full pipeline:
- Loads config and data
- Preprocesses and frames dataset
- Tunes and validates models
- Trains final model with selected HPs
- Evaluates and logs results

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