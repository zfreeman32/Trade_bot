
import pandas as pd
import numpy as np

# Dollar-Cost Averaging Strategy
def dollar_cost_averaging_signals(stock_df, investment_amount=100, investment_frequency='monthly'):
    """
    This function implements a Dollar-Cost Averaging (DCA) strategy, where a fixed amount is invested regularly.
    
    Parameters:
    stock_df (DataFrame): DataFrame containing stock price data with a 'Close' column.
    investment_amount (float): The amount of money to invest at each interval.
    investment_frequency (str): The frequency of investments ('monthly', 'weekly', etc.).
    
    Returns:
    DataFrame: A DataFrame containing 'Date', 'Investment', and 'Total Shares' columns indicating investment actions.
    """
    
    investment_frequency_map = {
        'monthly': 'M',
        'weekly': 'W',
        # Future enhancements could allow for daily or custom frequencies.
    }
    
    if investment_frequency not in investment_frequency_map:
        raise ValueError("Unsupported investment frequency. Choose either 'monthly' or 'weekly'.")
    
    resampled_data = stock_df['Close'].resample(investment_frequency_map[investment_frequency]).first()
    investment_data = pd.DataFrame(index=resampled_data.index)
    
    investment_data['Investment'] = investment_amount
    investment_data['Price'] = resampled_data
    investment_data['Shares Bought'] = investment_data['Investment'] / investment_data['Price']
    
    # Calculate the total shares owned over time
    investment_data['Total Shares'] = investment_data['Shares Bought'].cumsum()
    
    return investment_data[['Investment', 'Price', 'Total Shares']]
