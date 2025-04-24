
import pandas as pd
import numpy as np

# Standard Error Bands Strategy
def standard_error_bands_signals(stock_df, window_regression=21, window_sma=3, num_std_errors=2):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the linear regression
    x = np.arange(window_regression)
    y = stock_df['Close'].rolling(window=window_regression).apply(lambda prices: np.polyfit(x, prices, 1)[0] * x + np.polyfit(x, prices, 1)[1], raw=True)
    
    # Calculate standard error of the regression
    residuals = stock_df['Close'].rolling(window=window_regression).apply(lambda prices: np.sqrt(np.mean((prices - np.polyfit(x, prices, 1)[0] * x - np.polyfit(x, prices, 1)[1]) ** 2)), raw=True)
    
    # Calculate the middle line (3-period SMA of regression line)
    signals['Middle_Line'] = y.rolling(window=window_sma).mean()
    
    # Calculate the upper and lower bands
    signals['Upper_Band'] = signals['Middle_Line'] + num_std_errors * residuals.rolling(window=window_sma).mean()
    signals['Lower_Band'] = signals['Middle_Line'] - num_std_errors * residuals.rolling(window=window_sma).mean()

    # Generate signals
    signals['Signal'] = 'neutral'
    signals.loc[(stock_df['Close'] > signals['Upper_Band']), 'Signal'] = 'short'
    signals.loc[(stock_df['Close'] < signals['Lower_Band']), 'Signal'] = 'long'
    
    return signals[['Signal']]
