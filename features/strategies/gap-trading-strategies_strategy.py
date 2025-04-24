
import pandas as pd

# Gap Trading Strategy
def gap_trading_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    signals['Open'] = stock_df['Open']
    
    # Calculate the gap
    signals['Gap'] = signals['Open'] - signals['Close'].shift(1)
    
    # Generate signals based on the gap
    signals['gap_signal'] = 'neutral'
    
    # Long signal: Buy if the gap is down and the price moves up
    signals.loc[(signals['Gap'] < 0) & (signals['Close'] > signals['Open']), 'gap_signal'] = 'long'
    
    # Short signal: Sell if the gap is up and the price moves down
    signals.loc[(signals['Gap'] > 0) & (signals['Close'] < signals['Open']), 'gap_signal'] = 'short'
    
    return signals[['gap_signal']]
