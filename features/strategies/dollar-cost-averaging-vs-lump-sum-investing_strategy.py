
import pandas as pd

# Dollar Cost Averaging vs Lump Sum Strategy
def dca_vs_lump_sum_signals(stock_df, dca_investment=250, lump_sum_investment=3000, investment_duration_years=10, months_per_year=12):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Assume we invest lump-sum at the start of the period
    lump_sum_result = lump_sum_investment * (stock_df['Close'] / stock_df['Close'].iloc[0])
    
    # Implement Dollar Cost Averaging for each month
    total_investments = investment_duration_years * months_per_year
    monthly_investment = dca_investment
    
    # Create a cumulative investment value for DCA
    dca_cumulative = []
    for i in range(total_investments):
        if i < len(stock_df):
            dca_value = sum(
                [monthly_investment * (stock_df['Close'].iloc[j] / stock_df['Close'].iloc[0]) for j in range(i + 1)]
            )
            dca_cumulative.append(dca_value)
        else:
            dca_cumulative.append(None)
    
    signals['lump_sum_value'] = lump_sum_result
    signals['dca_value'] = dca_cumulative
    
    # Generate signals based on performance comparison
    signals['signal'] = 'neutral'
    signals.loc[signals['dca_value'] > signals['lump_sum_value'], 'signal'] = 'dca_superior'
    signals.loc[signals['dca_value'] < signals['lump_sum_value'], 'signal'] = 'lump_sum_superior'
    
    return signals
