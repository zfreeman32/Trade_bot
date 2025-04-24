
import pandas as pd

# Meb Faber Tactical Asset Allocation Strategy
def meb_faber_tactical_allocation_signals(asset_data):
    signals = pd.DataFrame(index=asset_data.index)
    
    # Assuming asset_data is a DataFrame with columns: ['US_Stocks', 'Foreign_Stocks', 'US_Bonds', 'US_REITs', 'World_Commodities']
    # Calculate the moving averages for each asset class
    signals['US_Stocks_MA'] = asset_data['US_Stocks'].rolling(window=200).mean()
    signals['Foreign_Stocks_MA'] = asset_data['Foreign_Stocks'].rolling(window=200).mean()
    signals['US_Bonds_MA'] = asset_data['US_Bonds'].rolling(window=200).mean()
    signals['US_REITs_MA'] = asset_data['US_REITs'].rolling(window=200).mean()
    signals['World_Commodities_MA'] = asset_data['World_Commodities'].rolling(window=200).mean()
    
    # Determine market conditions, 1 for bullish (price above MA), 0 for neutral
    signals['US_Stocks_Signal'] = (asset_data['US_Stocks'] > signals['US_Stocks_MA']).astype(int)
    signals['Foreign_Stocks_Signal'] = (asset_data['Foreign_Stocks'] > signals['Foreign_Stocks_MA']).astype(int)
    signals['US_Bonds_Signal'] = (asset_data['US_Bonds'] > signals['US_Bonds_MA']).astype(int)
    signals['US_REITs_Signal'] = (asset_data['US_REITs'] > signals['US_REITs_MA']).astype(int)
    signals['World_Commodities_Signal'] = (asset_data['World_Commodities'] > signals['World_Commodities_MA']).astype(int)
    
    # Determine overall signals -- all need to be 1 for a long signal, all 0 for a neutral
    signals['Overall_Signal'] = (signals['US_Stocks_Signal'] + 
                                  signals['Foreign_Stocks_Signal'] + 
                                  signals['US_Bonds_Signal'] + 
                                  signals['US_REITs_Signal'] + 
                                  signals['World_Commodities_Signal'])
    
    # Define the trading signals
    signals['Position'] = 'neutral'
    signals.loc[signals['Overall_Signal'] == 5, 'Position'] = 'long'  # all assets bullish
    signals.loc[signals['Overall_Signal'] == 0, 'Position'] = 'short' # all assets bearish

    return signals[['Position']]
