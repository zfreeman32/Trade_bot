
import pandas as pd
import numpy as np
from ta import trend

# Linear Regression Indicator Strategy
def linear_regression_signals(stock_df, window=14):
    signals = pd.DataFrame(index=stock_df.index)
    linear_reg = trend.LinearRegressionIndicator(stock_df['Close'], window)
    signals['Linear_Regression'] = linear_reg.lin_reg()
    
    signals['signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['Linear_Regression']), 'signal'] = 'long'
    signals.loc[(stock_df['Close'] < signals['Linear_Regression']), 'signal'] = 'short'
    
    return signals[['signal']]
