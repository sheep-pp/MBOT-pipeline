import tkinter as tk
from tkinter import filedialog, simpledialog

def select_data_file():
    root = tk.Tk()
    root.withdraw()  
    robot_file = filedialog.askopenfilename(title = 'select the robot data file')
    root.destroy()
    return robot_file

def select_emg_file():
    root = tk.Tk()
    root.withdraw()  
    emg_file = filedialog.askopenfilename(title = 'select the emg matlab structure ')
    root.destroy()
    return emg_file

def select_video_file():
    root = tk.Tk()
    root.withdraw()  
    video_file = filedialog.askopenfilename(title = 'select the corresponding video') 
    return video_file

def select_DLC_file():
    root = tk.Tk()
    root.withdraw()
    dlc_file = filedialog.askopenfilename(title = 'select the corresponding DLC file')
    root.destroy()
    return dlc_file

def specify_json_file_path():
    root = tk.Tk()
    root.withdraw() 
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])  
    return file_path 