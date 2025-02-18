import pandas as pd
from ta import volatility, candlestick

def BollingerBandsWithEngulfing(stock_df, length=20, num_dev=2, atr_factor=3, reward_risk_ratio=1.0):
    
    # Calculate Bollinger Bands using ta
    bb = volatility.BollingerBands(close=stock_df['Close'], window=length, window_dev=num_dev)
    stock_df['upper_band'] = bb.bollinger_hband()
    stock_df['mid_band'] = bb.bollinger_mavg()
    stock_df['lower_band'] = bb.bollinger_lband()
    
    # Calculate ATR using ta
    atr = volatility.AverageTrueRange(high=stock_df['High'], low=stock_df['Low'], close=stock_df['Close'], window=length)
    stock_df['ATR'] = atr.average_true_range()
    
    # Find Bullish Engulfing using ta
    stock_df['bullish_engulfing'] = ta.candlestick.CDL_ENGULFING(stock_df['Open'], stock_df['High'], stock_df['Low'], stock_df['Close'])

    # Set the initial value of all signals to neutral
    stock_df['BollingerBandsWithEngulfing_signals'] = 'neutral'
    
    # Generate signals
    for i in range(length, len(stock_df)):
        
        # Check for bullish engulfing pattern
        if stock_df['bullish_engulfing'].iloc[i] > 0:
            
            # Check if close price is above the lower band
            if stock_df['Close'].iloc[i] > stock_df['lower_band'].iloc[i]:
                
                # Calculate risk reward ratio
                reward = stock_df['upper_band'].iloc[i] - stock_df['High'].iloc[i]
                risk = stock_df['High'].iloc[i] - (stock_df['Close'].iloc[i] - atr_factor * stock_df['ATR'].iloc[i])
                
                # Check if risk-reward ratio is above the minimum
                if (risk != 0) and (reward / risk > reward_risk_ratio):
                    # Buy signal
                    stock_df.loc[stock_df.index[i], 'BollingerBandsWithEngulfing_signals'] = 'buy'
        
        # Exit criteria
        if stock_df['BollingerBandsWithEngulfing_signals'].iloc[i-1] == 'buy':
            # If high price exceeds upper band or low price falls below the stop price
            if stock_df['High'].iloc[i] > stock_df['upper_band'].iloc[i] or stock_df['Low'].iloc[i] < (stock_df['Close'].iloc[i] - atr_factor * stock_df['ATR'].iloc[i]):
                # Sell signal
                stock_df.loc[stock_df.index[i], 'BollingerBandsWithEngulfing_signals'] = 'sell'
    
    return stock_df[['BollingerBandsWithEngulfing_signals']]

#%% Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is in datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = BollingerBandsWithEngulfing(stock_df)

# Display output
print(signals_df.head())
