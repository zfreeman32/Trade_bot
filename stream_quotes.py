import tkinter as tk
from tkinter import ttk
import robin_stocks as r
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

# Log in to Robinhood
r.robinhood.login("zeb.freeman2000@gmail.com", "Zrocket32$$")

# Initialize empty DataFrame for SPY data
spy_data = pd.DataFrame(columns=['close_price', 'volume', 'time'])

def update_data():
    global spy_data
    global update_interval
    global plot_canvas
    
    # Get current quote for SPY
    spy = r.robinhood.get_quotes('SPY')
    
    # Append new data to DataFrame
    new_data = {'close_price': float(spy[0]['last_trade_price']),
                'volume': int(spy[0]['ask_size']),
                'time': time.strftime('%H:%M:%S')}
    # Append new data to DataFrame
    new_data_df = pd.DataFrame([new_data])
    spy_data = pd.concat([spy_data, new_data_df], ignore_index=True)

    
    # Update labels with new data
    close_price_label.config(text=f"Close Price: {new_data['close_price']}")
    volume_label.config(text=f"Volume: {new_data['volume']}")
    time_label.config(text=f"Last Updated: {new_data['time']}")
    
    # Plot the data
    ax.clear()
    ax.plot(spy_data['time'], spy_data['close_price'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Close Price')
    plot_canvas.draw()

    # Schedule the next update
    root.after(update_interval, update_data)

# Create GUI
root = tk.Tk()
root.title("Stock Data")

# Labels
close_price_label = tk.Label(root, text="Close Price: ")
close_price_label.pack()

volume_label = tk.Label(root, text="Volume: ")
volume_label.pack()

time_label = tk.Label(root, text="Last Updated: ")
time_label.pack()

# Create Matplotlib figure and canvas
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(1, 1, 1)
plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas.get_tk_widget().pack()

# Update interval in milliseconds (1 minute)
update_interval = 60000

# Start updating data
update_data()

# Run the GUI main loop
root.mainloop()
