
import pandas as pd

# Double Down Trading Strategy
def double_down_signals(stock_df, initial_investment, num_shares):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Position'] = 0  # Indicates whether we are in a trade or not
    signals['Price'] = stock_df['Close']
    signals['Average_Cost'] = 0
    signals['Action'] = 'neutral'
    
    for i in range(1, len(stock_df)):
        # Check if the price drops from the last price
        if stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i - 1]:
            if signals['Position'].iloc[i - 1] == 0:
                # Initial buy
                signals['Position'].iloc[i] = num_shares
                signals['Average_Cost'].iloc[i] = stock_df['Close'].iloc[i]
                signals['Action'].iloc[i] = 'buy'
            else:
                # Double down (buy additional shares)
                signals['Position'].iloc[i] = signals['Position'].iloc[i - 1] + num_shares
                signals['Average_Cost'].iloc[i] = (signals['Average_Cost'].iloc[i - 1] * signals['Position'].iloc[i - 1] + stock_df['Close'].iloc[i] * num_shares) / signals['Position'].iloc[i]
                signals['Action'].iloc[i] = 'double_down'

        elif stock_df['Close'].iloc[i] > signals['Average_Cost'].iloc[i - 1]:
            # If the price recovers above the average cost, we sell
            signals['Action'].iloc[i] = 'sell'
            signals['Position'].iloc[i] = 0  # Exit the position
            signals['Average_Cost'].iloc[i] = 0
        
        else:
            # Hold the position
            signals['Position'].iloc[i] = signals['Position'].iloc[i - 1]
            signals['Average_Cost'].iloc[i] = signals['Average_Cost'].iloc[i - 1]
    
    return signals[['Action', 'Position', 'Average_Cost']]
