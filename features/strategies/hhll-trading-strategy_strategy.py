
import pandas as pd

def hhll_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['Price'] = stock_df['Close']
    signals['Signal'] = 'neutral'
    
    # Setting initial values for tracking last lower low and higher high
    last_lower_low = None
    last_higher_high = None
    
    for i in range(1, len(stock_df)):
        # Detect a lower low
        if stock_df['Close'][i] < stock_df['Close'][i - 1]:
            if last_lower_low is None or stock_df['Close'][i] < last_lower_low:
                last_lower_low = stock_df['Close'][i]

        # Detect a higher high
        if stock_df['Close'][i] > stock_df['Close'][i - 1]:
            if last_higher_high is None or stock_df['Close'][i] > last_higher_high:
                last_higher_high = stock_df['Close'][i]

        # Check for buying condition (downtrend reversal)
        if last_lower_low is not None and last_higher_high is not None:
            if stock_df['Close'][i] >= last_lower_low and stock_df['Close'][i - 1] < last_lower_low:
                signals['Signal'][i] = 'long'
            # Check for selling condition (uptrend reversal)
            elif stock_df['Close'][i] <= last_higher_high and stock_df['Close'][i - 1] > last_higher_high:
                signals['Signal'][i] = 'short'

    return signals
