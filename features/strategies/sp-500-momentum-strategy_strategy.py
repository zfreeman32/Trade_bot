
import pandas as pd

# S&P 500 Momentum Strategy
def sp500_momentum_signals(stock_df, lookback_period=6, rebalance_frequency='M'):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the returns over the lookback period
    stock_df['Returns'] = stock_df['Close'].pct_change(periods=lookback_period)
    
    # Assign a "long" signal for the top performing stocks based on returns
    signals['signal'] = 'neutral'
    top_performers = stock_df['Returns'].nlargest(10).index  # Top 10 performing stocks
    
    signals.loc[top_performers, 'signal'] = 'long'
    
    # Rebalance monthly
    signals['rebalance'] = signals.index.to_period(rebalance_frequency).drop_duplicates()
    
    # Generate signals based on rebalancing
    for period in stock_df['rebalance'].unique():
        mask = signals['rebalance'] == period
        current_returns = stock_df.loc[mask]['Returns']
        top_performers = current_returns.nlargest(10).index
        signals.loc[mask, 'signal'] = 'neutral'
        signals.loc[top_performers, 'signal'] = 'long'
    
    return signals[['signal']]
