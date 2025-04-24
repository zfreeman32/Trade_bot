
import pandas as pd

# Martingale Trading Strategy
def martingale_signals(stock_df, initial_investment=1, max_trades=10):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    investment = initial_investment
    position_size = initial_investment

    for i in range(1, len(stock_df)):
        # Check if the previous trade resulted in a loss
        if signals['signal'].iloc[i - 1] == 'short' and stock_df['Close'].iloc[i] > stock_df['Close'].iloc[i - 1]:
            # If previous position was losing, double the position size
            position_size *= 2
            signals.at[signals.index[i], 'signal'] = 'short'
        elif signals['signal'].iloc[i - 1] == 'long' and stock_df['Close'].iloc[i] < stock_df['Close'].iloc[i - 1]:
            # If previous position was losing for long, double the position size
            position_size *= 2
            signals.at[signals.index[i], 'signal'] = 'long'
        else:
            # Resolve the position, we have to assume the market changed direction
            trade_result = (stock_df['Close'].iloc[i] - stock_df['Close'].iloc[i - 1]) * (1 if signals['signal'].iloc[i - 1] == 'long' else -1)

            if trade_result > 0:  # successful trade
                signals.at[signals.index[i], 'signal'] = 'neutral'
                position_size = initial_investment  # reset position size after a win
            else:  # unsuccessful trade
                # Only allow for a maximum number of trades to avoid infinite losses
                if investment < max_trades:
                    investment += 1
                    signals.at[signals.index[i], 'signal'] = 'neutral'  # still in the losing trade
                else:
                    signals.at[signals.index[i], 'signal'] = 'neutral'  # stop trading after max trades

    return signals
