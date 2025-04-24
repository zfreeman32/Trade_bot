
import pandas as pd

# Bogleheads 3 and 4 Fund Portfolio Strategy
def bogleheads_portfolio_signals(stock_df, allocation_3=[0.60, 0.20, 0.20], allocation_4=[0.50, 20.0, 20.0, 10.0]):
    signals = pd.DataFrame(index=stock_df.index)
    total_investment = stock_df['Close'] * stock_df['Volume']

    # Assuming stock_df has columns ['Close', 'Volume']
    # Calculate total assets based on allocations for the 3 Fund Portfolio
    signals['3_Fund_Investment'] = total_investment * allocation_3[0] + total_investment * allocation_3[1] + total_investment * allocation_3[2]

    # Generate signals based on simple conditions
    signals['3_Fund_Signal'] = 'neutral'
    signals.loc[signals['3_Fund_Investment'].pct_change() > 0, '3_Fund_Signal'] = 'long'
    signals.loc[signals['3_Fund_Investment'].pct_change() < 0, '3_Fund_Signal'] = 'short'

    # Calculate total assets based on allocations for the 4 Fund Portfolio
    signals['4_Fund_Investment'] = total_investment * allocation_4[0] + total_investment * allocation_4[1] + total_investment * allocation_4[2] + total_investment * allocation_4[3]

    # Generate signals based on simple conditions for 4 fund portfolio
    signals['4_Fund_Signal'] = 'neutral'
    signals.loc[signals['4_Fund_Investment'].pct_change() > 0, '4_Fund_Signal'] = 'long'
    signals.loc[signals['4_Fund_Investment'].pct_change() < 0, '4_Fund_Signal'] = 'short'

    return signals[['3_Fund_Signal', '4_Fund_Signal']]
