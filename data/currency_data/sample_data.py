import pandas as pd
#%%
from collections import deque

input_file = r"C:\Users\zebfr\Documents\All_Files\TRADING\Trading_Bot\currency_data\EURUSD_full_1min.txt"
output_file = "sampled_EURUSD_1min.txt"
num_lines = 2_000_000  # Adjust as needed

with open(input_file, "r") as f:
    last_lines = deque(f, num_lines)  # Efficiently stores only the last N lines

with open(output_file, "w") as f:
    f.writelines(last_lines)  # Write to a new file

# %%
import pandas as pd

input_file = "sampled_EURUSD_1min.txt"
output_file = "sampled_EURUSD_1min.csv"

# Define column names
columns = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]

# Read the text file into a DataFrame
df = pd.read_csv(input_file, names=columns)

# Save as CSV
df.to_csv(output_file, index=False)

print(f"CSV file saved as {output_file}")

# %%
