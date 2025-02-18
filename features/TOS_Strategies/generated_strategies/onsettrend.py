import pandas as pd

def onset_trend_detector(stock_df, k1=0.8, k2=0.4):
    # Placeholder implementations
    def super_smoother_filter(data):
        pass  # Replace with actual implementation
    def roofing_filter(data):
        pass  # Replace with actual implementation
    def quotient_transform(data, k):
        pass  # Replace with actual implementation

    signals = pd.DataFrame(index=stock_df.index)
    
    # Apply Super Smoother Filter and Roofing Filter
    filtered_price = roofing_filter(super_smoother_filter(stock_df['Close']))
    
    # Compute oscillators using quotient transform
    oscillator1 = quotient_transform(filtered_price, k1)
    oscillator2 = quotient_transform(filtered_price, k2)
    
    # Generate signals
    signals['oscillator1'] = oscillator1
    signals['oscillator2'] = oscillator2
    signals['onset_trend_detector_signals'] = 'neutral'
    signals.loc[(signals['oscillator1'] > 0) & (signals['oscillator1'].shift(1) <= 0), 'onset_trend_detector_signals'] = 'long'
    signals.loc[(signals['oscillator2'] < 0) & (signals['oscillator2'].shift(1) >= 0), 'onset_trend_detector_signals'] = 'short'
    
    signals.drop(['oscillator1', 'oscillator2'], axis=1, inplace=True)
    
    return signals
