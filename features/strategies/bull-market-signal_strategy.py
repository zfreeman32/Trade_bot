
import pandas as pd

# Bull Market Signal Strategy
def bull_market_signal(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    stock_df['200_SMA'] = stock_df['Close'].rolling(window=200).mean()

    # Identify bear market: when the price drops 20% from the last all-time high
    stock_df['All_Time_High'] = stock_df['Close'].cummax()
    stock_df['Bear_Market'] = stock_df['Close'] < (stock_df['All_Time_High'] * 0.8)

    # Calculate if there are 18 consecutive closes above the 200 SMA
    stock_df['Above_200_SMA'] = stock_df['Close'] > stock_df['200_SMA']
    signals['Bull_Signal'] = stock_df['Above_200_SMA'].rolling(window=18).sum() == 18

    # Generate trading signals
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['Bull_Signal'] & stock_df['Bear_Market'], 'trade_signal'] = 'long'
    
    return signals[['trade_signal']]
