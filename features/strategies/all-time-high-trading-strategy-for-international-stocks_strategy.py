
import pandas as pd
from ta import momentum

# Trading strategy based on quantified patterns and anomalies
def quantified_strategy_signals(stock_df, window=14):
    """Generates trading signals based on quantified patterns and abnormal returns.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing stock price data with a 'Close' column.
    window (int): The window size for calculating momentum.
    
    Returns:
    pd.DataFrame: DataFrame containing trading signals ('long', 'short', 'neutral').
    """
    signals = pd.DataFrame(index=stock_df.index)
    
    # Calculate the momentum indicator
    momentum_indicator = momentum.StochasticOscillator(stock_df['High'], stock_df['Low'], stock_df['Close'], window=window)
    signals['momentum'] = momentum_indicator.stoch()
    
    # Initialize the signals as 'neutral'
    signals['strategy_signal'] = 'neutral'
    
    # Generate buy and sell signals based on momentum
    signals.loc[(signals['momentum'] > 80), 'strategy_signal'] = 'short'  # Overbought condition, potential shorting opportunity
    signals.loc[(signals['momentum'] < 20), 'strategy_signal'] = 'long'   # Oversold condition, potential buying opportunity
    
    return signals[['strategy_signal']]
