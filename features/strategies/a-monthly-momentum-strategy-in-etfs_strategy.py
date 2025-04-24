
import pandas as pd

# Monthly Momentum Strategy for EEM, SPY, and TLT
def monthly_momentum_signals(price_df):
    signals = pd.DataFrame(index=price_df.index)
    returns = price_df.pct_change(periods=1).fillna(0)  # Monthly returns
    
    # Calculate cumulative returns to determine momentum
    cumulative_returns = (1 + returns).cumprod()
    
    # Determine the ETF with the highest momentum at the end of each month
    signals['selected_etf'] = cumulative_returns.apply(lambda x: x.idxmax(), axis=1)
    
    # Generate long signals based on the selected ETF
    signals['signal'] = 'neutral'
    signals.loc[signals['selected_etf'] == 'EEM', 'signal'] = 'long_EEM'
    signals.loc[signals['selected_etf'] == 'SPY', 'signal'] = 'long_SPY'
    signals.loc[signals['selected_etf'] == 'TLT', 'signal'] = 'long_TLT'
    
    # Drop the selected_etf column, returning only the signal
    signals.drop(['selected_etf'], axis=1, inplace=True)
    
    return signals
