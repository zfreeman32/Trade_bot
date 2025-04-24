
import pandas as pd
import numpy as np

# Covered Calls Trading Strategy
def covered_calls_signals(stock_df, strike_price, option_expiry_days=30):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Price'] = stock_df['Close']
    signals['Position'] = 'none'

    # Generate buy signals based on price movement and holding a long position
    signals.loc[signals['Price'] < strike_price, 'Position'] = 'long'  # Buy and hold
    
    # Generate covered call signal by selling call options
    # The logic considers selling an option close to the current price
    signals.loc[(signals['Position'] == 'long') & 
                 (stock_df['Close'] >= strike_price), 'Position'] = 'sell_call'

    # Assuming a weekly expiry option cycle for simplicity in this strategy
    # Marking neutral position after the expiry
    signals['Position'] = np.where(signals['Position'] == 'sell_call', 'neutral', signals['Position'])
    
    # Forward fill to ensure the position holds until the option expiry
    signals['Position'] = signals['Position'].ffill()
    
    return signals[['Price', 'Position']]
