
import pandas as pd

# Disparity Index Trading Strategy
def disparity_index_signals(stock_df, moving_average_period=20, threshold=5):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Close'] = stock_df['Close']
    
    # Calculating the moving average
    signals['Moving_Average'] = stock_df['Close'].rolling(window=moving_average_period).mean()
    
    # Calculating the Disparity Index
    signals['Disparity_Index'] = ((signals['Close'] - signals['Moving_Average']) / signals['Moving_Average']) * 100
    
    # Initializing signals
    signals['di_signal'] = 'neutral'
    
    # Long signal: when the Disparity Index exceeds the threshold
    signals.loc[signals['Disparity_Index'] > threshold, 'di_signal'] = 'long'
    
    # Short signal: when the Disparity Index falls below the negative threshold
    signals.loc[signals['Disparity_Index'] < -threshold, 'di_signal'] = 'short'
    
    return signals[['Close', 'Moving_Average', 'Disparity_Index', 'di_signal']]
