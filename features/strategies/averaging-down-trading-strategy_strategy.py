
import pandas as pd

# Averaging Down Trading Strategy
def averaging_down_signals(stock_df, initial_buy_price, additional_buy_price, position_size=100):
    """
    Implements the Averaging Down trading strategy.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    initial_buy_price (float): The price at which the initial shares were bought.
    additional_buy_price (float): The price at which additional shares are bought.
    position_size (int): The number of shares to buy initially and additionally.
    
    Returns:
    pd.DataFrame: DataFrame containing trading signals based on the averaging down strategy.
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate average price after first buy
    total_invested = initial_buy_price * position_size
    avg_price_after_first_buy = total_invested / position_size
    
    # Calculate average price after second buy
    total_invested += additional_buy_price * position_size
    avg_price_after_second_buy = total_invested / (position_size * 2)
    
    signals['avg_price'] = avg_price_after_second_buy
    signals['signal'] = 'neutral'
    
    # Generate buy signal when the price is below the average price after the second buy
    signals.loc[stock_df['Close'] < avg_price_after_second_buy, 'signal'] = 'buy'
    
    # Generate sell signal when the price rises above the average price after the second buy
    signals.loc[stock_df['Close'] > avg_price_after_second_buy, 'signal'] = 'sell'
    
    return signals[['signal']]
