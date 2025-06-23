# Feature Selection Summary and Rationale

## Selection Methodology

The optimal feature set was derived from comprehensive analysis of four different trading datasets spanning multiple time periods and including both technical indicators and trading strategies. The selection prioritizes features that demonstrate:

1. **Consistency across multiple analyses**
2. **High predictive power (correlation with target)**
3. **Temporal stability**
4. **Diverse signal types to avoid redundancy**

## Core Technical Indicators (High Consistency & Performance)

### **EFI (Ease of Movement Index)**
- **Selected in all 4 analyses**
- **Highest SHAP importance** in recent analyses (0.29, 0.17, 0.21)
- Captures price-volume relationship effectively
- Strong correlation with profitable signals (0.051-0.057)

### **z_score (Price Normalization)**
- **Selected in all 4 analyses**
- **Consistently high SHAP importance** (0.20, 0.12, 0.22)
- Excellent correlation with target (0.075-0.076)
- High temporal stability across time periods

### **CCI (Commodity Channel Index)**
- **Selected in 3 out of 4 analyses**
- Strong correlation with target (0.071-0.071)
- Good stability metrics (CV < 0.01)
- Effective momentum oscillator

### **TRIX (Triple Exponential Average)**
- **Selected in 3 out of 4 analyses**
- High stability across time periods
- Good SHAP importance (0.06-0.08)
- Trend-following indicator with noise reduction

## High-Impact Volume & Price Features

### **PLUS_DM (Plus Directional Movement)**
- **Highest correlation with target** (0.106-0.114)
- Selected in multiple analyses
- Key component of ADX system
- Strong directional trend signal

### **Volume**
- **Highest correlation in strategy analysis** (0.102)
- **Critical for signal validation**
- High SHAP importance (0.098)
- Fundamental market participation metric

### **current_candle_height & average_candle_height**
- **High correlation** (0.099 and 0.075 respectively)
- Price action volatility measures
- Important for short-term signal generation
- Selected in strategy-focused analysis

## Momentum & Trend Indicators

### **ROCP/ROCR100 (Rate of Change)**
- **Good correlation** (0.078) and stability
- Selected in multiple analyses
- Momentum measurement
- Price velocity indicator

### **RSI (Relative Strength Index)**
- **High importance in strategy analysis** (0.075 feature importance)
- Classic overbought/oversold indicator
- Good correlation with target (0.071)
- Widely used in trading systems

### **MINUS_DI (Minus Directional Indicator)**
- Selected in multiple analyses
- Negative correlation (-0.049 to -0.050)
- Bearish trend strength measurement
- Complements PLUS_DM

### **STOCH_D (Stochastic %D)**
- Selected in multiple analyses
- Good correlation (0.048-0.049)
- Momentum oscillator
- Effective for timing entries

## Volatility & Market Structure

### **ATR (Average True Range)**
- **Good correlation** (0.075) with target
- Selected in technical analysis
- Volatility measurement
- Risk assessment indicator

### **rolling_std (Rolling Standard Deviation)**
- High correlation (0.083) in original analysis
- Volatility proxy
- Market uncertainty measure

## Hilbert Transform Features (Cycle Analysis)

### **HT_DCPHASE, HT_PHASOR_INPHASE, HT_DCPERIOD**
- Selected across multiple analyses
- Cycle and phase relationship indicators
- Advanced signal processing techniques
- Market timing components

## Strategy Signals (High Predictive Power)

### **bb_short_entry_signal (Bollinger Bands Short Entry)**
- **Highest mutual information** (0.162)
- Mean reversion strategy signal
- High negative correlation (-0.088)

### **cci_bearish_signal**
- **High mutual information** (0.118)
- Momentum-based bearish signal
- Good correlation (-0.079)

### **rsi_overbought_signal**
- Good correlation (0.081)
- Classic overbought condition
- High-probability reversal signal

### **williams_sell_signal & stochrsi_overbought_signal**
- Additional momentum-based signals
- Diversify signal sources
- Good correlation (0.079 and 0.069)

## Final Feature Set Rationale

The selected 25 features provide:

1. **Diversity**: Volume, momentum, trend, volatility, and strategy signals
2. **Consistency**: Features selected across multiple time periods and datasets
3. **Performance**: High correlation with profitable signals
4. **Stability**: Low coefficient of variation across time splits
5. **Complementarity**: Different aspects of market behavior without redundancy

This balanced approach ensures robust model performance while maintaining interpretability and avoiding overfitting to specific market conditions.