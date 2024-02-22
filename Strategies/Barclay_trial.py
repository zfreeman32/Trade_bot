#%%
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate annualized volatility
def calculate_volatility(returns):
    return np.sqrt(252) * returns.std()

# Function to calculate sector realized volatility
def calculate_sector_realized_volatility(stock_data, sector_data):
    return calculate_volatility(stock_data) - calculate_volatility(sector_data)

#%%
# Ticker symbol of the stock
ticker = 'AAPL'
# Start and end dates for the data
start_date = '2023-01-01'
end_date = '2024-01-01'

# Download stock price data
stock_price_data = yf.download(ticker, start=start_date, end=end_date)
# Calculate daily returns
stock_returns = stock_price_data['Adj Close'].pct_change().dropna()

# Assuming sector_data is another stock for the sector
sector_ticker = 'SPY'
sector_price_data = yf.download(sector_ticker, start=start_date, end=end_date)
sector_returns = sector_price_data['Adj Close'].pct_change().dropna()
print (stock_price_data)
print (yf.Ticker(ticker).info)
#%%

prices = yf.Ticker(ticker).history(period='1mo')
prices.head()
#%% 
# Calculate implied volatility (IV)
stock_iv = stock_price_data['IV']

# Calculate historical volatility (HV)
stock_hv = calculate_volatility(stock_returns)

# Calculate sector realized volatility
sector_realized_volatility = calculate_sector_realized_volatility(stock_returns, sector_returns)

# Plot the values on a chart
plt.figure(figsize=(10, 6))
plt.plot(stock_iv, label='Implied Volatility (IV)')
plt.axhline(y=stock_hv, color='r', linestyle='--', label='Historical Volatility (HV)')
plt.axhline(y=sector_realized_volatility, color='g', linestyle='--', label='Sector Realized Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.title('Implied Volatility, Historical Volatility, and Sector Realized Volatility')
plt.legend()
plt.show()

# %%
