
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
csv_file = r'C:\Users\zeb.freeman\Documents\Trade_bot\currency_data\EURUSD.csv'
data = pd.read_csv(csv_file)

# Convert the data to a Pandas DataFrame
data = pd.DataFrame(data).reset_index(drop=True)

# Slice the DataFrame to include only the first 100 rows
data_subset = data.loc[1050:1400]

# Plot the 'Date' against 'Close' for the first 100 data points
# Create the plot
plt.figure(figsize=(12, 6))  # Set the figure size
plt.plot(data_subset['Date'], data_subset['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('SPY Stock Close Price Over Time (First 100 Data Points)')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()