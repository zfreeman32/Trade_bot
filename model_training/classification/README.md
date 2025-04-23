# Deep Learning Classification Pipeline for Financial Signals

This project builds a scalable, feature-rich deep learning pipeline for binary classification on financial time series data. It is specifically designed for predicting trading signals (e.g., short/long entries) using technical indicators and custom-engineered features.

---

## 📈 Problem Overview

The task is to classify each timestep into a binary decision (e.g., `short_signal` = 0 or 1) based on a wide range of engineered features derived from market data (EUR/USD 1-min). The pipeline supports large-scale model experimentation with multiple neural architectures.

---

## 🛠️ Key Features & Workflow

### 1. **Feature Engineering**
- Raw data is cleaned using domain-specific preprocessing
- Volume-based features undergo log transformation, winsorization, and percentile ranking
- Lag features are generated for both the target and indicators like RSI, CCI, EFI, etc.
- Adds rolling statistics (mean, std) over lagged targets

### 2. **Feature Selection**
- Important features loaded from a prior feature importance study
- Only selected features and signal columns are retained

### 3. **Sliding Window View**
- Converts tabular data into supervised format for sequential models
- Input = [n_timesteps x n_features], Target = [signal at t+1]

### 4. **Data Scaling**
- Uses `RobustScaler` with clipping to handle outliers and stabilize learning
- Prevents NaNs, infinities, and constant features through robust preprocessing

### 5. **Class Imbalance Handling**
- Class weights computed dynamically from training set
- Focal loss is introduced to address imbalance during model training

---

## 🧠 Supported Architectures
The pipeline supports multiple deep learning models via modular builders:

- LSTM / GRU / Conv1D / Conv1D+LSTM
- BiLSTM with Attention
- Transformer
- MultiStream Hybrid
- Temporal Convolution Network (TCN)
- ResNet-based classifier

> Builders are passed as dictionary objects, allowing dynamic looping and tuning.

---

## 🧪 Training Methodology

### Progressive Training
- **Phase 1**: Train on 20% of training data (subset) to warm up weights
- **Phase 2**: Fine-tune on full training set using lower LR
- Reduces overfitting and stabilizes convergence

### Hyperparameter Tuning
- Uses `keras-tuner` with **Bayesian Optimization**
- Explores learning rate, focal loss parameters (alpha/gamma), etc.

### Cross-Validation
- Uses `TimeSeriesSplit` for **ordered folds**
- Evaluates on val_loss, val_accuracy, val_auc, etc.
- Skips models with too many NaN validation results

---

---

## 📊 Evaluation Metrics

- `accuracy`, `precision`, `recall`, `AUC`
- Per-fold and final test set metrics
- Handles instability (NaNs, exploding gradients) with:
  - Gradient clipping
  - Learning rate warmup
  - Termination on NaN
