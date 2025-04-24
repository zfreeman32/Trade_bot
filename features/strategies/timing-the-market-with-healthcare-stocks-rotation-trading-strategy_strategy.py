
import pandas as pd

# Membership Plans Trading Strategy
def membership_plans_signals(stock_df):
    signals = pd.DataFrame(index=stock_df.index)
    signals['signal'] = 'neutral'
    
    # Example logic for trading signals based on membership plans
    # Assuming we are using a hypothetical monthly membership signal generator
    monthly_signal = stock_df['Close'].pct_change(periods=30)  # 30-day returns

    # Generate Long signals when the returns over 30 days exceed a certain threshold
    signals.loc[monthly_signal > 0.05, 'signal'] = 'long'
    # Generate Short signals when the returns over 30 days are below a certain threshold
    signals.loc[monthly_signal < -0.05, 'signal'] = 'short'
    
    return signals
