#%%
import sys
import pandas as pd
import all_Strategies
from concurrent.futures import ThreadPoolExecutor  # Using threads instead of processes

# Wrapper function for applying strategy
def apply_strategy(strategy_func, stock_df):
    return strategy_func(stock_df)

# Apply all strategies using ThreadPoolExecutor
def generate_all_signals(stock_df):
    strategy_functions = [
        all_Strategies.ppo_signals,
        all_Strategies.Awesome_Oscillator_signals,
        all_Strategies.kama_cross_signals,
        all_Strategies.williams_signals,
        all_Strategies.tsi_signals,
        all_Strategies.stoch_signals,
        all_Strategies.roc_signals,
        all_Strategies.rsi_signals,
        all_Strategies.stochrsi_signals,
        all_Strategies.cci_signals,
        all_Strategies.dpo_signals,
        all_Strategies.ema_signals,
        all_Strategies.ichimoku_signals,
        all_Strategies.kst_signals,
        all_Strategies.macd_signals,
        all_Strategies.golden_ma_signals,
        all_Strategies.short_ma_signals,
        all_Strategies.strategy_5_8_13,
        all_Strategies.keltner_channel_strategy,
        all_Strategies.cmf_signals,
        all_Strategies.eom_signals,
        all_Strategies.mfi_signals,
        all_Strategies.strategy_w5_8_13,
        all_Strategies.adx_strength_direction,  
        all_Strategies.mass_index_signals, 
        all_Strategies.psar_signals,
        all_Strategies.aroon_strategy, 
        all_Strategies.atr_signals, 
        all_Strategies.rsi_signals_with_divergence,
        all_Strategies.stc_signals, 
        all_Strategies.vortex_signals, 
        all_Strategies.golden_wma_signals,
        all_Strategies.short_wma_signals,  
        all_Strategies.donchian_channel_strategy,  
        all_Strategies.turnaround_tuesday_strategy,
        all_Strategies.ironbot_trend_filter,
        all_Strategies.frama_signals,
        all_Strategies.high_volume_points,
        all_Strategies.fractal_ema_signals
    ]

    # Use threading instead of multiprocessing
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda func: apply_strategy(func, stock_df), strategy_functions))

    # Merge results efficiently
    all_signals_df = pd.concat([stock_df] + results, axis=1)

    return all_signals_df
#%%
# # Load data and apply signals
# file_path = r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\sampled2k_EURUSD_1min.csv'
# stock_df = pd.read_csv(file_path)
# signals_df = generate_all_signals(stock_df)

# # Display output
# signals_df.head()
#%%