import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from features import call_Strategies
import features.all_indicators as all_indicators

# Generate indicators
def generate_features(data):
    df_with_indicators = all_indicators.generate_all_indicators(data)

    # Generate Signals
    df_strategies = pd.DataFrame(data).reset_index(drop=True)
    print(df_strategies.head())

    # Now pass the DataFrame to the generate_all_signals function
    df_strategies = call_Strategies.generate_all_signals(df_strategies)
    df_strategies = df_strategies.drop(['kama_signal'],axis=1)

    # Convert columns to string
    data['Date'] = data['Date'].astype(str)  
    data['Time'] = data['Time'].astype(str)
    data['datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%Y%m%d %H:%M:%S')

    # Identify common columns except 'Date'
    common_cols = [col for col in df_strategies.columns if col in df_with_indicators.columns and col != 'Date']

    # Remove common columns from df2 before merging
    df_with_indicators = df_with_indicators.drop(columns=common_cols)

    # Then do your merge
    df_strategies = df_strategies.set_index('Date')
    df_with_indicators = df_with_indicators.set_index('Date')
    merged_df = pd.concat([df_strategies, df_with_indicators], axis=1).reset_index()
    df = pd.DataFrame(merged_df)
    df = df.loc[:,~df.columns.duplicated()].copy()
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')  # Convert to datetime
    df['Minutes'] = (df['Time'] - df['Time'].min()).dt.total_seconds() / 60  
    df.drop(columns=['Time'], inplace=True)

    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Remove 'Time' and 'Date' from non_numeric_cols if they exist
    cols_to_encode = [col for col in non_numeric_cols if col not in ['Time']]

    # Initialize LabelEncoder
    label_encoders = {}
    for col in cols_to_encode:
        # Replace NaN values with 0
        df[col].fillna(0, inplace=True)
        df[col].replace("", 0, inplace=True)
        df[col] = df[col].fillna('').apply(str).str.strip() 
        print(df[col].shape) # Convert all values to string and strip whitespace
        le = LabelEncoder()
        df[col]=df[col].astype(str)
        df[col] = le.fit_transform(df[col]) 
        label_encoders[col] = le 

    df.fillna(-9999, inplace=True)
    return df