#%%
import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Generate indicators
def clean_data(data):
    # Generate Signals
    df = pd.DataFrame(data).reset_index(drop=True)
    print(df.head())

    def find_duplicate_columns(df):
        duplicate_cols = {}
        cols = df.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if df[cols[i]].equals(df[cols[j]]):
                    duplicate_cols[cols[i]] = cols[j]
        return duplicate_cols

    duplicate_cols = find_duplicate_columns(df)
    print('Duplicate Columns:\n', duplicate_cols)

    df = df.loc[:, ~df.T.duplicated()]

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

#%%
# import pandas as pd

# # File Path should be OHLCV Data
# file_path = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_1min_sampled_features.csv"

# # Read CSV into DataFrame
# df = pd.read_csv(file_path, header=0)
# strat_df = clean_data(df)
# strat_df
# %%
