
import pandas as pd

# Peter Lynch Investment Strategy
def lynch_investment_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['price_to_earnings'] = stock_df['Close'] / stock_df['Earnings']
    
    # Identifying 'ten baggers'
    signals['is_ten_bagger'] = stock_df['Close'] >= 10 * stock_df['Purchase_Price']
    
    # Value signal: P/E ratio under 15
    signals['value_signal'] = 'neutral'
    signals.loc[signals['price_to_earnings'] < 15, 'value_signal'] = 'long'
    
    # Growth signal: positive earnings growth
    signals['growth_signal'] = 'neutral'
    signals.loc[stock_df['Earnings_Growth'] > 0, 'growth_signal'] = 'long'
    
    # Combining signals
    signals['final_signal'] = 'neutral'
    signals.loc[(signals['value_signal'] == 'long') & (signals['growth_signal'] == 'long'), 'final_signal'] = 'long'
    signals.loc[(signals['is_ten_bagger'] == True), 'final_signal'] = 'long'
    
    return signals[['value_signal', 'growth_signal', 'final_signal']]
