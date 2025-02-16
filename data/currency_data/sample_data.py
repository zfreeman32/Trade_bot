#%%
import pandas as pd

#%%
# File paths
input_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\data\currency_data\EURUSD_full_1min.txt"
output_file = "sampled_EURUSD_1min.csv"
num_lines = 2_000_000  # Number of lines to sample

# Read file efficiently using tail() from pandas
df = pd.read_csv(input_file, names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"], engine="python")

# Take only the last N rows
df = df.tail(num_lines)
df

#%%
# Save as CSV
df.to_csv(output_file, index=False)
print(f"CSV file saved as {output_file}")
