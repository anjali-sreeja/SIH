'''
import cv2
import os
import csv
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def mediapipe_detection(frame, model):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
   # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def record_landmarks(results, action):
    try:
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

        right_hand = results.right_hand_landmarks.landmark
        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())

        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
 
        row = pose_row + right_hand_row + left_hand_row
        row.insert(0, action)

        with open('coords.csv', mode='a', newline='') as f:
            csv_writer_obj = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer_obj.writerow(row)
            
    except:
        pass
'''

primary_color = "#393646"
secondary_color = "#F4EEE0"
accent_color = "#4F4557"
underHover_color = "#8f8a7f"

from tkinter import *
from detection import start_detection
root = Tk()
root.geometry("500x400")
root.configure(bg=primary_color)

padx1 = 10
pady1 = 6

options_frame = Frame(root, bg=accent_color)

def hide_indicate():
    home_indicate.config(bg=accent_color)
    add_indicate.config(bg=accent_color)
    about_indicate.config(bg=accent_color)
    contact_indicate.config(bg=accent_color)

def delete_pages():
    for frame in main_frame.winfo_children():
        frame.destroy()

def show_indicate(lb, page):
    hide_indicate()
    lb.config(bg=secondary_color)
    delete_pages()
    page()
home_btn = Button(options_frame, text="Home", font=("Helvetica", "16"), fg=secondary_color, bd=0, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, command=lambda: show_indicate(home_indicate, home_page))
home_btn.place(x=10, y=50)

home_indicate = Label(options_frame, text=" ", background=accent_color)
home_indicate.place(x=3, y=50, width=5, height=40)

add_btn = Button(options_frame, text="Add Sign", font=("Helvetica", "16"), fg=secondary_color, bd=0, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, command=lambda: show_indicate(add_indicate, add_page))
add_btn.place(x=10, y=100)

add_indicate = Label(options_frame, text=" ", background=accent_color)
add_indicate.place(x=3, y=100, width=5, height=40)

about_btn = Button(options_frame, text="About", font=("Helvetica", "16"), fg=secondary_color, bd=0, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, command=lambda: show_indicate(about_indicate, about_page))
about_btn.place(x=10, y=150)

about_indicate = Label(options_frame, text=" ", background=accent_color)
about_indicate.place(x=3, y=150, width=5, height=40)

contact_btn = Button(options_frame, text="Contact", font=("Helvetica", "16"), fg=secondary_color, bd=0, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, command=lambda: show_indicate(contact_indicate, contact_page))
contact_btn.place(x=10, y=200)

contact_indicate = Label(options_frame, text=" ", background=accent_color)
contact_indicate.place(x=3, y=200, width=5, height=40)


options_frame.pack(side=LEFT)
options_frame.pack_propagate(False)
options_frame.configure(height=400, width=125)



##################################     MAIN         #######################
main_frame = Frame(root, bg=primary_color)


def home_page():
    home_frame = Frame(main_frame)

    homeTop_frame = Frame(home_frame, bg=primary_color)
    homeBottom_frame = Frame(home_frame, bg=primary_color)
    
    heading = Label(homeTop_frame, text="SignSense", background=secondary_color, font=("Helvetica", "28"), foreground=secondary_color, bg=primary_color, padx=200)
    sub_heading = Label(homeTop_frame, text="Real-Time Sign Language Recognition", background=secondary_color, font=("Helvetica", "14"), padx=100, foreground=secondary_color, bg=primary_color)
    
    button_frame = Frame(homeBottom_frame, bg=primary_color, padx=140)
    detect_btn = Button(homeBottom_frame, text="Detect", command=detect_func,  font=("Helvetica", "16"), fg=secondary_color, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, padx=15, pady=4)
    
    skeleton = IntVar()
    skeleton_checkbox = Checkbutton(homeBottom_frame, onvalue=1, offvalue=0, text="Skeletal Mode", variable=skeleton, font=("Helvetica", "14"), fg="#4C6E83" ,bg=primary_color, activebackground=primary_color)
    
    
    heading.pack()
    sub_heading.pack()
    detect_btn.pack()
    button_frame.pack()
    skeleton_checkbox.pack()
    
    homeTop_frame.pack(side=TOP)
    homeTop_frame.pack_propagate(False)
    homeTop_frame.configure(height=250, width=500)

    homeBottom_frame.pack(side=BOTTOM)
    homeBottom_frame.pack_propagate(False)
    homeBottom_frame.configure(height=200, width=500)

    home_frame.pack(pady=20)



def add_page():
    add_frame = Frame(main_frame)

    addTop_frame = Frame(add_frame, bg=primary_color)
    addBottom_frame = Frame(add_frame, bg=primary_color)
    
    #heading = Label(homeTop_frame, text="SignSense", background=secondary_color, font=("Helvetica", "28"), foreground=secondary_color, bg=primary_color, padx=200)
    sub_heading = Label(addTop_frame, text="Add in your own Signs and Gestures", background=secondary_color, font=("Helvetica", "16"), padx=100, foreground=secondary_color, bg=primary_color)

    instruction = Label(addTop_frame, text="Instructions:\n\n1. Train the machine by providing 50 video sequences\n\n2. Each Video Sequence consist of 30 frames\n\n3. Repeat your action/sign every sequence\n\n4. A two seconds gap is given between\nevery sequences for the user to reposition", background=secondary_color, font=("Helvetica", "11"), padx=100, foreground=secondary_color, bg=primary_color, pady=25)
    
    button_frame = Frame(addBottom_frame, bg=primary_color, padx=140)
    input_frame = Frame(addBottom_frame, bg=primary_color, padx=20)
    add_btn = Button(addBottom_frame, text="New Sign", command=detect_func,  font=("Helvetica", "16"), fg=secondary_color, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, padx=15, pady=4)
    
    skeleton = IntVar()
    #skeleton_checkbox = Checkbutton(addBottom_frame, onvalue=1, offvalue=0, text="Skeletal Mode", variable=skeleton, font=("Helvetica", "14"), fg="#4C6E83" ,bg=primary_color, activebackground=primary_color)
    action_input_field = Entry(input_frame, width=25, font=("Helvetica", "12"))
    name_text = Label(input_frame, text="Name: ", background=secondary_color, font=("Helvetica", "12"), padx=5, foreground=secondary_color, bg=primary_color)

    
    # heading.pack()
    sub_heading.pack()
    instruction.pack()
    name_text.grid(row=0, column=0)
    action_input_field.grid(row=0, column=1)
    input_frame.pack()
    #input_frame.pack_propagate(False)
    add_btn.pack(pady=20)
    button_frame.pack()
    #skeleton_checkbox.pack()
    
    addTop_frame.pack(side=TOP)
    addTop_frame.pack_propagate(False)
    addTop_frame.configure(height=250, width=500)

    addBottom_frame.pack(side=BOTTOM)
    addBottom_frame.pack_propagate(False)
    addBottom_frame.configure(height=200, width=500)

    add_frame.pack(pady=20)


def about_page():
    about_frame = Frame(main_frame)

    aboutTop_frame = Frame(about_frame, bg=primary_color)
    aboutBottom_frame = Frame(about_frame, bg=primary_color)
    
    heading = Label(aboutTop_frame, text="About", background=secondary_color, font=("Helvetica", "28"), foreground=secondary_color, bg=primary_color, padx=200)
    text = Label(aboutBottom_frame, text="SignSense is a real-time hand sign detection\n software designed to empower users in effective\n communication through sign language.\n Leveraging advanced computer vision and \nmachine learning, the software captures \nlive video input, tracks hand movements,\n and recognizes diverse hand signs in real-time. \n\n\nThis software is made as a prototype for SIH 2023", background=secondary_color, font=("Helvetica", "13"), padx=100,pady=10 ,foreground=secondary_color, bg=primary_color)

    #button_frame = Frame(aboutBottom_frame, bg=primary_color, padx=140)
    #detect_btn = Button(aboutBottom_frame, text="Detect", command=detect_func,  font=("Helvetica", "16"), fg=secondary_color, bg=accent_color, activeforeground=underHover_color, activebackground=accent_color, padx=15, pady=4)
    
    #heckbox = Checkbutton(aboutBottom_frame, onvalue=1, offvalue=0, text="Skeletal Mode", variable=skeleton, font=("Helvetica", "14"), fg="#4C6E83" ,bg=primary_color, activebackground=primary_color)
    
    
    heading.pack()
    text.pack()
    #detect_btn.pack()
    # button_frame.pack()
    # skeleton_checkbox.pack()
    
    aboutTop_frame.pack(side=TOP)
    aboutTop_frame.pack_propagate(False)
    aboutTop_frame.configure(height=80, width=500)
    
    aboutBottom_frame.pack(side=BOTTOM)
    aboutBottom_frame.pack_propagate(False)
    aboutBottom_frame.configure(height=200, width=500)

    
    about_frame.pack(pady=20)


def contact_page():
    contact_frame = Frame(main_frame)
    
    sub_heading = Label(contact_frame, text="For any queries related to the software,\nPlease contact\n\nPh no: 8130337161, 8618079805\n\nEmail: karthiksureshnair6@gmail.com", background=secondary_color, font=("Helvetica", "14"), padx=100, pady=85, foreground=secondary_color, bg=primary_color)
    
    skeleton = IntVar()
    # skeleton_checkbox = Checkbutton(homeBottom_frame, onvalue=1, offvalue=0, text="Skeletal Mode", variable=skeleton, font=("Helvetica", "14"), fg="#4C6E83" ,bg=primary_color, activebackground=primary_color)
    
    
    # heading.pack()
    sub_heading.pack()
    # detect_btn.pack()
    # button_frame.pack()
    # skeleton_checkbox.pack()



    contact_frame.pack(pady=20)



main_frame.pack(side=RIGHT)
main_frame.pack_propagate(False)
main_frame.configure(height=400, width=500)

def detect_func():
    start_detection()

def add_func():
    print("added")


show_indicate(home_indicate, home_page)



root.mainloop()