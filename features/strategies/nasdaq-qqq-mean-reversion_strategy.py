
import pandas as pd

# Nasdaq QQQ Mean Reversion Trading Strategy
def qqq_mean_reversion_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['close'] = stock_df['Close']
    signals['high_yesterday'] = stock_df['High'].shift(1)
    signals['close_yesterday'] = stock_df['Close'].shift(1)

    # Initialize the signal column
    signals['signal'] = 'neutral'

    # Generate long signals: current close is below both yesterday's high and yesterday's close
    signals.loc[(signals['close'] < signals['high_yesterday']) & (signals['close'] < signals['close_yesterday']), 'signal'] = 'long'

    # Generate exit signals: current close is above yesterday's high or above yesterday's close
    signals.loc[(signals['close'] > signals['high_yesterday']) | (signals['close'] > signals['close_yesterday']), 'signal'] = 'exit'

    return signals[['signal']]
