# create_dataset/main.py

import tkinter as tk
from tkinter import ttk, messagebox
from data_capture import capture_data

def start_data_capture():
    action_name = action_entry.get()
    collection_time = int(time_var.get())
    if action_name.strip() == '':
        messagebox.showwarning("Input Error", "Please enter a valid action name.")
        return
    root.destroy()
    capture_data(action_name, collection_time)

# Create main window
root = tk.Tk()
root.title("Action Data Collection")
root.geometry("400x200")  # Adjust window size

# Action name input
action_label = tk.Label(root, text="Enter action name:")
action_label.pack(pady=5)
action_entry = tk.Entry(root, width=30)
action_entry.pack(pady=5)

# Collection time dropdown
time_label = tk.Label(root, text="Select collection time (seconds):")
time_label.pack(pady=5)
time_options = [10, 30, 60, 90, 120]
time_var = tk.StringVar(value="30")
time_dropdown = ttk.Combobox(root, textvariable=time_var, values=time_options, state='readonly', width=27)
time_dropdown.pack(pady=5)

# Confirm button
confirm_button = tk.Button(root, text="Confirm", command=start_data_capture)
confirm_button.pack(pady=20)

root.mainloop()
