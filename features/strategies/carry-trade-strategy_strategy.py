
import pandas as pd

# Carry Trade Strategy
def carry_trade_signals(currency_df, low_rate_currency, high_rate_currency):
    """
    Generates trading signals for a carry trade strategy based on interest rates
    of two currencies.
    
    Parameters:
        currency_df (pd.DataFrame): Dataframe containing currency exchange rates.
        low_rate_currency (str): The currency with a low interest rate.
        high_rate_currency (str): The currency with a high interest rate.
        
    Returns:
        pd.DataFrame: A dataframe containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=currency_df.index)
    
    # Calculate the return from the high-rate currency
    signals['high_rate_return'] = currency_df[high_rate_currency].pct_change()
    
    # Calculate the return from the low-rate currency (which we would borrow)
    signals['low_rate_return'] = currency_df[low_rate_currency].pct_change()
    
    # Calculate net carry return (the difference in returns)
    signals['carry_diff'] = signals['high_rate_return'] - signals['low_rate_return']
    
    # Generate signals based on carry return
    signals['trade_signal'] = 'neutral'
    signals.loc[signals['carry_diff'] > 0, 'trade_signal'] = 'long'  # Buy high rate currency
    signals.loc[signals['carry_diff'] < 0, 'trade_signal'] = 'short'  # Sell high rate currency
    
    return signals[['trade_signal']]
