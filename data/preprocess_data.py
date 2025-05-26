import sys
sys.path.append(r'C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats

def clean_data(data):
    # Generate Signals
    df = pd.DataFrame(data).reset_index(drop=True)
    print(df.head())

    def find_and_remove_duplicate_columns_efficient(df):
        """
        More efficient way to find and remove duplicate columns
        """
        print("Finding duplicate columns...")
        
        # Method 1: Use column hashes for initial filtering (much faster)
        column_hashes = {}
        columns_to_keep = []
        duplicate_cols = {}
        
        for col in df.columns:
            # Create a hash of the column values
            col_hash = hash(tuple(df[col].values))
            
            if col_hash in column_hashes:
                # Potential duplicate found, verify with equals()
                original_col = column_hashes[col_hash]
                if df[col].equals(df[original_col]):
                    duplicate_cols[col] = original_col
                    print(f"Duplicate found: {col} == {original_col}")
                else:
                    # Hash collision but not actually duplicate
                    columns_to_keep.append(col)
                    # Create a new hash key to avoid future collisions
                    column_hashes[f"{col_hash}_{col}"] = col
            else:
                column_hashes[col_hash] = col
                columns_to_keep.append(col)
        
        print(f'Duplicate Columns: {duplicate_cols}')
        
        # Keep only non-duplicate columns
        df_cleaned = df[columns_to_keep]
        print(f"Removed {len(df.columns) - len(df_cleaned.columns)} duplicate columns")
        print(f"Original columns: {len(df.columns)}, After deduplication: {len(df_cleaned.columns)}")
        
        return df_cleaned, duplicate_cols

    # Use the efficient duplicate removal
    df, duplicate_cols = find_and_remove_duplicate_columns_efficient(df)

    # Identify non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    print(f"Non-numeric columns found: {len(non_numeric_cols)}")

    # Remove datetime-related columns from encoding
    datetime_cols = ['Time', 'Date', 'datetime', 'Datetime', 'DATETIME']
    cols_to_encode = [col for col in non_numeric_cols if col not in datetime_cols]
    print(f"Columns to encode: {len(cols_to_encode)} - {cols_to_encode}")

    # Initialize LabelEncoder with progress tracking
    label_encoders = {}
    for i, col in enumerate(cols_to_encode, 1):
        print(f"Encoding column {i}/{len(cols_to_encode)}: {col}")
        
        # Efficient preprocessing
        df[col] = df[col].fillna('0').astype(str).str.strip()
        
        # Only encode if there are multiple unique values
        unique_vals = df[col].nunique()
        if unique_vals > 1:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            # If only one unique value, just convert to 0
            df[col] = 0
            print(f"  Column {col} has only one unique value, setting to 0")

    # Handle numeric columns efficiently
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print(f"Processing {len(numeric_cols)} numeric columns for missing values...")
    
    # Vectorized missing value handling
    for col in numeric_cols:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            print(f"  Filling {missing_count} missing values in {col}")
            # Use median for skewed distributions
            if abs(stats.skew(df[col].dropna())) > 1:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mean())
    
    print("Data cleaning completed!")
    return df