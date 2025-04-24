
import pandas as pd

# Warren Buffett ETF Portfolio (90/10) Strategy
def warren_buffett_etf_signals(stock_df, stock_allocation=0.9, bond_allocation=0.1):
    signals = pd.DataFrame(index=stock_df.index)
    
    # Define ETFs for stock and bonds
    stock_etf = 'SPY'  # S&P 500 ETF
    bond_etf = 'SHY'   # Short-term Government Bonds ETF
    
    # Define allocations
    total_investment = 100  # Assume we are starting with 100 units of currency for simplicity
    stock_investment = total_investment * stock_allocation
    bond_investment = total_investment * bond_allocation
    
    # Generate investment signals
    signals['investment_signal'] = 'neutral'
    
    # Long signal for stocks if the returns are positive and neutral otherwise
    stock_returns = stock_df['Close'].pct_change() * stock_investment
    if stock_returns.iloc[-1] > 0:
        signals['investment_signal'] = 'long'
    
    # Bond signal for bonds if the bond returns are positive
    bond_df = pd.DataFrame()  # Placeholder for bond data, should contain bond price data
    if not bond_df.empty:
        bond_returns = bond_df['Close'].pct_change() * bond_investment
        if bond_returns.iloc[-1] > 0:
            signals.loc[signals['investment_signal'] != 'long', 'investment_signal'] = 'long'
    
    return signals
