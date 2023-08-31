#%%
# Read in Library
import pandas as pd
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import pandas as pd
from ta import add_all_ta_features

#%%
# Read in Price Data
csv_file = 'SPY.csv'
stock_df = pd.read_csv(csv_file)

# Stock Financial Data

#%%
# Indicators
indicators_df = pd.DataFrame(index=stock_df.index)

# Add all technical indicators using TA library
indicators_df = add_all_ta_features(
    stock_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=False
)
print(indicators_df.columns)

#%%
# Strategies
# Awesome Oscillator
def Awesome_Oscillator_signals(stock_df):
    # Define buy and sell signals
    signals = pd.DataFrame(index=stock_df.index)
    signals['Ao_signal'] = 0  # Initialize all signals to 0

    for i in range(1, len(stock_df)):
        if (
            stock_df['momentum_ao'].iloc[i-1] < 0 and
            stock_df['momentum_ao'].iloc[i] >= 0
        ):
            signals['Ao_signal'].iloc[i] = 1  # Buy signal
        elif (
            stock_df['momentum_ao'].iloc[i-1] >= 0 and
            stock_df['momentum_ao'].iloc[i] < 0
        ):
            signals['Ao_signal'].iloc[i] = -1  # Sell signal
    return signals

ao_signals = pd.DataFrame(Awesome_Oscillator_signals(stock_df))
print(ao_signals)

# Analyst Rating and Events

#%% 
column_names = pd.DataFrame(indicators_df.columns)
# Pre-process Data


# Models

# Predict Buy/Sell Condition

# Evaluate

# %%
