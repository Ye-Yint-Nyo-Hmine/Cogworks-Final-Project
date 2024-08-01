import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from speech_button import SpeechButton
from LLM.TTS import recognize_speech_from_mic, speak

from LLM.main import run_chatbot, load_sys_prompt, initialize_profile
from LLM.utils.cryptographer import decrypt
from LLM.profiles.Profile import Profile
# from object_detection.detection import detect_camera as object_detection
# from face_recognition.main import recognize as face_recognition
# from fire_detection.main import detect_fire
# from baby_crying_detection.main import detect_baby_cry

def start_object_detection():
    # Start object detection in a new thread
    # Thread(target=object_detection).start()
    pass

def start_face_recognition():
    # Start face recognition in a new thread
    # Thread(target=face_recognition).start()
    pass

def start_fire_detection():
    # Start fire detection in a new thread
    # Thread(target=detect_fire).start()
    pass

def start_baby_cry_detection():
    # Start baby crying detection in a new thread
    # Thread(target=detect_baby_cry).start()
    pass

def send_message():
    
    """if user_message.strip() == "":
        return

    chat_history.insert(tk.END, "You: " + user_message + "\n")
    user_input.delete(0, tk.END)
    
    # Here you should call your LLM chatbot function and pass the user_message
    # response = run_llm_chatbot(user_message)
    response = "This is a placeholder response from the LLM."
    
    chat_history.insert(tk.END, "LLM: " + response + "\n")"""
    


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

## update history


"""object_detection_button = tk.Button(button_frame, text="Start Object Detection", command=start_object_detection)
object_detection_button.pack(side=tk.LEFT, padx=5)

face_recognition_button = tk.Button(button_frame, text="Start Face Recognition", command=start_face_recognition)
face_recognition_button.pack(side=tk.LEFT, padx=5)

fire_detection_button = tk.Button(button_frame, text="Start Fire Detection", command=start_fire_detection)
fire_detection_button.pack(side=tk.LEFT, padx=5)

baby_cry_detection_button = tk.Button(button_frame, text="Start Baby Cry Detection", command=start_baby_cry_detection)
baby_cry_detection_button.pack(side=tk.LEFT, padx=5)"""

# Start Tkinter main loop
root.mainloop()

