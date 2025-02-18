import pandas as pd
import numpy as np

def calculate_divergence(df_main, df_secondary, length=25):
    # Calculate regression divergence
    df_main['return'] = df_main.price.pct_change()
    df_secondary['return'] = df_secondary.price.pct_change()
    df_main['return'].rolling(window=length).corr(df_secondary['return'])
    return df_main['return']

def calculate_corr3(df_main, df_tertiary, roc_length):
    # Calculate correlation with tertiary symbol
    df_main['roc'] = df_main.price.pct_change(periods=roc_length)
    df_tertiary['roc'] = df_tertiary.price.pct_change(periods=roc_length)
    correlation = df_main['roc'].rolling(window=roc_length).corr(df_tertiary['roc'])
    return correlation

def regression_divergence_strat(df_main, df_secondary, df_tertiary, length=25, roc_length, divergence_momentum_length, exit_length, divergence_critical_level, corr3level):
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=df_main.index)
    signals['divergence'] = calculate_divergence(df_main, df_secondary, length)
    signals['corr3'] = calculate_corr3(df_main, df_tertiary, roc_length)

    # Buy to open
    signals['buy'] = (signals['divergence'] > divergence_critical_level) & (signals['divergence'].shift(1) <= divergence_critical_level) & (signals['corr3'] < corr3level)
    
    # Sell to open
    signals['sell'] = (signals['divergence'] < -divergence_critical_level) & (signals['divergence'].shift(1) >= -divergence_critical_level) & (signals['corr3'] < corr3level)

    # Exits
    signals['exit_buy'] = signals['buy'].shift(exit_length)
    signals['exit_sell'] = signals['sell'].shift(exit_length)

    signals = signals.dropna()

    return signals
#%%
# Load data and apply signals
file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
stock_df = pd.read_csv(file_path, index_col=0)

# Ensure index is datetime format
stock_df.index = pd.to_datetime(stock_df.index)

# Call function with correct arguments
signals_df = regression_divergence_strat(stock_df)

# Display output
print(signals_df.head())