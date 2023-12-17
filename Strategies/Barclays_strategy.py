#%%
from numpy import row_stack
import pandas as pd
import yfinance as yf  # You may need to install this library
from datetime import datetime, timedelta
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

#%%
# stocks = ['NVDA', 'AAPL', 'MSFT', 'ADBE', 'EBAY', 'NFLX', 'GOOGL', 'AMZN']

price_data_AMZN= yf.download('AMZN', start='2020-11-06', end='2023-1-3', auto_adjust = True)
price_data_AMZN

#%%
# Compute the logarithmic returns using the Closing price
price_data_AMZN['Log_Ret'] = np.log(price_data_AMZN['Close'] / price_data_AMZN['Close'].shift(1))

# Compute Volatility using the pandas rolling standard deviation function
price_data_AMZN['Realized Volatility'] = price_data_AMZN['Log_Ret'].rolling(window=252).std() * np.sqrt(252)

#%%
# Plot the AMZN Price series and the Volatility
price_data_AMZN[['Close']].plot(subplots=True, color='blue',figsize=(8, 6))
plt.title('Close price', color='purple', size=15)
# Setting axes labels for close prices plot
plt.xlabel('Dates', {'color': 'orange', 'fontsize':15})
plt.ylabel('Prices', {'color': 'orange', 'fontsize':15})
price_data_AMZN[['Realized Volatility']].plot(subplots=True, color='blue',figsize=(8, 6))
plt.title('Realized Volatility', color='purple', size=15)
plt.xlabel('Dates', {'color': 'orange', 'fontsize':15})
plt.ylabel('Realized volatility', {'color': 'orange', 'fontsize':15})
plt.xticks(rotation=45)

def calculate_realized_volatility(price_data, window_size=20):
    returns = price_data.pct_change().dropna()
    volatility = returns.rolling(window=window_size).std() * np.sqrt(252)  # Assuming 252 trading days in a year

    return volatility

# Function to calculate VolScore
def calculate_vol_score(stock_data):
    return row['ImpliedVolatility'] - (row['ImpliedVolatility'] - row['RealizedVolatility']) - row['SectorRealizedVolatility']

# Function to select stocks based on VolScore criteria
def select_stocks(data):
    selected_stocks = data[(data['ImpliedVolatility'] > data['RealizedVolatility']) &
                           (data['ImpliedVolatility'] > data['SectorRealizedVolatility'])]

    return selected_stocks

# Function to execute short straddle positions
def execute_short_volatility_strategy(selected_stocks):
    # Define option expiration date (e.g., 30 days from today)

    expiration_date = datetime.now() + timedelta(days=30)
    expiration_str = expiration_date.strftime('%Y-%m-%d')

    # Loop through selected stocks and execute short straddle positions
    for symbol in selected_stocks['Symbol']:
        try:
            # Fetch option chain data for the selected stock
            option_chain = yf.Ticker(symbol).option_chain(expiration_str)

            # Assuming a simplified approach: sell at-the-money call and put options
            at_the_money_strike = int(np.round(option_chain.calls['strike'].iloc[len(option_chain.calls) // 2]))

            # Example: Execute short call
            sell_call_order = execute_sell_to_open(symbol, expiration_str, at_the_money_strike, 'call')

            # Example: Execute short put
            sell_put_order = execute_sell_to_open(symbol, expiration_str, at_the_money_strike, 'put')

            print(f"Short straddle executed for {symbol} with strike {at_the_money_strike}.")
            
            # You would handle order execution details based on your trading platform or API
            # For demonstration purposes, assume the function execute_sell_to_open handles order execution
        except Exception as e:
            print(f"Error executing short straddle for {symbol}: {e}")

# Example function for executing sell-to-open orders
def execute_sell_to_open(symbol, expiration, strike, option_type):
    # Placeholder function; replace with actual execution logic based on your trading platform
    # Example: execute sell-to-open order using your preferred trading API or platform
    return f"Sell-to-open order for {symbol} - {expiration} - {option_type} - Strike {strike}"

# Example usage
if __name__ == "__main__":
    # Fetch historical price data
    symbol = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Calculate VolScore
    calculate_vol_score(stock_data)

    # Select stocks based on VolScore criteria
    selected_stocks = select_stocks(stock_data)

    # Execute short volatility strategy
    execute_short_volatility_strategy(selected_stocks)

#%%
from py_vollib.black_scholes_merton.implied_volatility import implied_volatility
price = 15.1
s = 113.06
k = 100.0
t = 14/365.0
r = 0.035 #3.5% 10 year treasury rate<br>
q = 0
flag = 'c'
iv = implied_volatility(price, s, k, t, r, q, flag)
print(iv)
# %%
