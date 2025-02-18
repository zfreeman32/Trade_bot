import pandas as pd
import numpy as np
from ta.momentum import StochasticOscillator
import yfinance as yf
from scipy.stats import linregress

def calculate_slope(series, periods):
    return linregress(range(len(series)), series)[0]

def get_three_period_divergence(data, dvg_length1=5, dvg_length2=10, dvg_length3=15, momentum_length=14):
    stochastic_oscillator = StochasticOscillator(data['High'], data['Low'], data['Close'], momentum_length)
    
    data['FastK'] = stochastic_oscillator.stoch()
    data['FastD'] = stochastic_oscillator.stoch_signal()
    
    data['price_slope1'] = data['Close'].rolling(dvg_length1).apply(calculate_slope)
    data['price_slope2'] = data['Close'].rolling(dvg_length2).apply(calculate_slope)
    data['price_slope3'] = data['Close'].rolling(dvg_length3).apply(calculate_slope)

    data['fastK_slope1'] = data['FastK'].rolling(dvg_length1).apply(calculate_slope)
    data['fastK_slope2'] = data['FastK'].rolling(dvg_length2).apply(calculate_slope)
    data['fastK_slope3'] = data['FastK'].rolling(dvg_length3).apply(calculate_slope)
    
    data['entry_signal'] = np.where((np.sign(data['price_slope1']) != np.sign(data['fastK_slope1'])) &
                                    (np.sign(data['price_slope2']) != np.sign(data['fastK_slope2'])) &
                                    (np.sign(data['price_slope3']) != np.sign(data['fastK_slope3'])), 1, 0)
    
    data['exit_signal'] = np.where((np.sign(data['price_slope1']) == np.sign(data['fastK_slope1'])) &
                                   (np.sign(data['price_slope2']) == np.sign(data['fastK_slope2'])) &
                                   (np.sign(data['price_slope3']) == np.sign(data['fastK_slope3'])), -1, 0)

    data['signal'] = data['entry_signal'] + data['exit_signal']
    return data

# Test the function
data = yf.download('AAPL', start='2020-01-01', end='2022-02-22')
signals = get_three_period_divergence(data)
#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = three_bar_inside_bar_se(stock_df)

# Display output
print(signals_df.head())