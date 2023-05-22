import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical

import pickle

from Detect import detect
from Generate_data import generate_dataset
from Trainning import add_train 
from Delete import delete

name_entry = None
name_label = None
label_mapping = {}

def update_label_mapping():
    global label_mapping
    with open('label_mapping.pkl', 'rb') as f:
        label_mapping = pickle.load(f)

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def getName1():
    global name_entry, name_label
    if name_label is not None:
        name_label.destroy()
        name_entry.destroy()
    name_label = tk.Label(frame_bottom, text="Name", font=("Algerian", 20))
    name_label.pack()
    name_entry = tk.Entry(frame_bottom, width=50, bd=5) #border
    name_entry.pack()
    name_entry.bind("<Return>", add_face)

def getName2():
    global name_entry, name_label
    if name_label is not None:
        name_label.destroy()
        name_entry.destroy()
    name_label = tk.Label(frame_bottom, text="Name", font=("Algerian", 20))
    name_label.pack()
    name_entry = tk.Entry(frame_bottom, width=50, bd=5) #border
    name_entry.pack()
    name_entry.bind("<Return>", pdelete)

def add_face(event=None):
    global name_entry
    name = name_entry.get() 
    if not name:  
        messagebox.showwarning("Warning", "Vui lòng nhập tên") 
    else:
        if name in label_mapping:
            result = messagebox.askyesno("Confirmation", f"Tên: '{name}' đã tồn tại, bạn có muốn ghi đè dữ liệu?")
            if result:
                generate_train(name)
            else:
                name_entry.delete(0, tk.END)
        else:
            generate_train(name)

        update_label_mapping()

def generate_train(name):
    generate_dataset(name)

    messagebox = tk.Toplevel(window)
    messagebox.geometry("300x100")
    messagebox.attributes("-topmost", True)
    message_label = tk.Label(messagebox, text="Đang thực hiện, vui lòng đợi...")
    message_label.pack(pady=20)

    add_train(name, label_mapping)

    window.after(500, messagebox.destroy)

def pdelete(event=None):
    global name_entry
    name = name_entry.get() 
    if not name:  
        messagebox.showwarning("Warning", "Vui lòng nhập tên") 
    else:
        check = delete(name, label_mapping)
        if check:
            messagebox.showinfo("Thông báo", "Xóa thành công")
            update_label_mapping()
        else:
            messagebox.showinfo("Thông báo", f"Tên: {name} không có trong tập dữ liệu")

def pdetect():
    detect(label_mapping)

window = tk.Tk()
window.title("Face recognition system")

update_label_mapping()

frame_top = tk.Frame(window)
frame_top.pack(pady=50)

frame_bottom = tk.Frame(window)
frame_bottom.pack(pady=10)

b1 = tk.Button(frame_top, text="Detect the face", font=("Algerian", 20), bg='green', fg='white', command=pdetect)
b1.pack(side="left", padx=10)

b2 = tk.Button(frame_top, text="Add face", font=("Algerian", 20), bg='pink', fg='black', command=getName1)
b2.pack(side="left", padx=10)

b3 = tk.Button(frame_top, text="Delete face", font=("Algerian", 20), bg='orange', fg='red', command=getName2)
b3.pack(side="left", padx=10)

window.geometry("800x400")
window.update_idletasks()
center_window(window)
window.mainloop()