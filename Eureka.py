#%%
# Read in Library

import pandas as pd
from ta import add_all_ta_features
from Strategies import call_Strategies

#%%
# Read in Price Data
csv_file = './data/SPY.csv'
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

all_signals_df = call_Strategies.generate_all_signals('./data/SPY.csv', './data/VIX.csv')
print(all_signals_df)

# True Signals (The most Optimal Buy/Sell Points since 1993)
true_signals_df = pd.read_csv("./data/SPY_true_signals.csv")

# Analyst Rating and Events

#%% 
# Pre-process Data
merge_df = pd.concat([indicators_df, all_signals_df, true_signals_df], axis = 1)

# Models

# Predict Buy/Sell Condition

# Evaluate

# %%
