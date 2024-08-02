import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from speech_button import SpeechButton
from LLM.TTS import recognize_speech_from_mic, speak

from LLM.main import run_chatbot, load_sys_prompt, initialize_profile
from LLM.utils.cryptographer import decrypt
from LLM.profiles.Profile import Profile
from object_detection.main import detect_camera
from motion_facial_recognition import main


"""
Instruction on implementation:

Inside your main program loop, import report_event function from report.py
Then, call the function when a report should be delivered:
eg:

while true:
    if detect_something == True:
        report_event("Describe the detection in anyway")
        detect_something = False
    
That's it!

Now make it run at the start of the interface by adding it inside one of the start_something() functions below using threading,
eg:

def start_something():
    Thread(target=foo).start()


start_something() 

"""
def start_object_detection():
    Thread(target=detect_camera).start() # example on how the thread should be like
    

def start_face_recognition():
    Thread(target=main).start()
    
def start_fire_detection():
    pass

def start_baby_cry_detection():
    pass


    
def update_report():
    prev = ""
    while True:
        with open("current_reports.txt") as f:
            read = f.read()
            f.close()
            if read != prev:
                speech_button.add_to_context(read)
                prev = read
        

color_palette = {
    "bg": "black",
    "text": "white",
    "accent": "aqua",
}


# Initialize Tkinter window
root = tk.Tk()
root.title("Safety Bytes")
root.config(bg=color_palette["bg"])







speech_button = SpeechButton(root, size=300, command=recognize_speech_from_mic, username="User") # Change with name
speech_button.place(x=600, y=10)

report_thread = Thread(target=update_report).start()



root.mainloop()

