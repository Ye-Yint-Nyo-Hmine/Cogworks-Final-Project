import tkinter as tk
import math
from threading import Thread
import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import queue
from LLM.main import initialize_profile, load_sys_prompt
from LLM.utils.cryptographer import decrypt
from LLM.TTS import speak

class SpeechButton(tk.Canvas):
    def __init__(self, master, size=100, command=None, username=None, custom_instr=False, **kwargs):
        super().__init__(master, width=size, height=size, background="black", highlightthickness=0, **kwargs)
        self.size = size
        self.angle1 = 0
        self.angle2 = 90
        self.angle3 = 180
        self.angle4 = 270
        self.is_pressed = False
        self.command = command
        self.output_queue = queue.Queue()
        self.chat_thread = None
        self.button_thread = Thread(target=self.run_command)
        
                # non host only
        self.load_sys_prompts = load_sys_prompt()
        self.KEY = "TLJRVBFrZY-0YVoflosslIcv8volnQTdSl940c"
        self.sys_prompt = decrypt(self.load_sys_prompts, self.KEY)


        self.template = """ 
        {sys_prompt}

        User profile:
        {user_profile}

        Here is the conversation history: 
        {context}


        User: {question}
        """


        
        self.username = username
        self.custom_instr = custom_instr
        
        
        self.model = OllamaLLM(model="llama3")
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.model
        
        
        
        if custom_instr:
            self.profile = initialize_profile(self.username, custom_instr[0], custom_instr[1])
        else:
            self.profile = initialize_profile(username)
        
        self.context = self.profile.try_load_history()
        
        

        # Draw the initial button
        self.create_oval(10, 10, size+20, size+20, fill="", outline="black", width=1)
        self.arc1 = self.create_arc(20, 20, size-10, size-30, start=0, extent=90, outline="green", style=tk.ARC, width=4)
        self.arc2 = self.create_arc(15, 15, size-26, size-6, start=90, extent=90, outline="green", style=tk.ARC, width=4)
        self.arc3 = self.create_arc(10, 10, size-6, size-26, start=180, extent=90, outline="green", style=tk.ARC, width=4)
        self.arc4 = self.create_arc(5, 5, size-30, size-10, start=270, extent=90, outline="green", style=tk.ARC, width=4)

        # Bind the button press event
        self.bind("<Button-1>", self.on_press)
        
    def run_command(self):
            if self.command:
                result = self.command()  # Run the command function
                self.output_queue.put(result)  

    def on_press(self, event):
        self.is_pressed = not self.is_pressed
        if self.is_pressed:
            self.start_animation()
            if self.command != None:
                self.button_thread.start()
                
        
        

    def start_animation(self):
        if self.is_pressed:
            speed = 5
            self.angle1 = (self.angle1 + speed) % 360
            color1 = self.angle_to_color(self.angle1)
            self.itemconfig(self.arc1, start=self.angle1, outline=color1)
            
            self.angle2 = (self.angle2 + speed + 10) % 360
            color2 = self.angle_to_color(self.angle2)
            self.itemconfig(self.arc2, start=self.angle2, outline=color2)
            
            self.angle3 = (self.angle3 + speed + 5) % 360
            color3 = self.angle_to_color(self.angle3)
            self.itemconfig(self.arc3, start=self.angle3, outline=color3)
            
            self.angle4 = (self.angle4 + speed + 15) % 360
            color4 = self.angle_to_color(self.angle4)
            self.itemconfig(self.arc4, start=self.angle4, outline=color4)
            
            self.after(50, self.start_animation)
            
            if not self.output_queue.empty():
                self.stop_animation()
    

    def angle_to_color(self, angle):
        # Create a gradient from red to yellow to blue
        angle = angle % 360
        if angle < 120:
            # Red to Yellow
            r = 255
            g = int((angle / 120) * 255)
            b = 0
        elif angle < 240:
            # Yellow to Blue
            r = int(((240 - angle) / 120) * 255)
            g = 255
            b = int(((angle - 120) / 120) * 255)
        else:
            # Blue to Red
            r = 0
            g = int(((360 - angle) / 120) * 255)
            b = 255

        return f"#{r:02x}{g:02x}{b:02x}"

    def stop_animation(self):
        self.is_pressed = False
        def slow_animation():
            speed = 3
            self.angle1 = (self.angle1 + speed) % 360
            color1 = self.angle_to_color(self.angle1)
            self.itemconfig(self.arc1, start=self.angle1, outline="aqua")
                
            self.angle2 = (self.angle2 + speed + 5) % 360
            color2 = self.angle_to_color(self.angle2)
            self.itemconfig(self.arc2, start=self.angle2, outline="aqua")
                
            self.angle3 = (self.angle3 + speed + 2.5) % 360
            color3 = self.angle_to_color(self.angle3)
            self.itemconfig(self.arc3, start=self.angle3, outline="aqua")
                
            self.angle4 = (self.angle4 + speed + 7) % 360
            color4 = self.angle_to_color(self.angle4)
            self.itemconfig(self.arc4, start=self.angle4, outline="aqua")
        print("doing something")
        self.after(50, slow_animation)
        if not self.output_queue.empty():
            result = self.output_queue.get()
            self.chat_thread = Thread(target=self.chatbot, args=[result])
            self.chat_thread.start()
            self.button_thread = Thread(target=self.run_command)
            
    def chatbot(self, user_input):
        print("You: " + user_input)
        if user_input.lower() == "exit":
            self.context += "\nUser: Exited ...\n\nUser has exited the chat. New session will began."
            self.profile.update_history(self.context)
            

        result = self.chain.invoke({"sys_prompt": self.sys_prompt,
                               "user_profile": self.profile.try_load_profile(),
                               "context": self.context, 
                               "question": user_input})
        self.itemconfig(self.arc1, start=self.angle1, outline="green")
        self.itemconfig(self.arc2, start=self.angle2, outline="green")
        self.itemconfig(self.arc3, start=self.angle3, outline="green")
        self.itemconfig(self.arc4, start=self.angle4, outline="green")
        print(f"ByteGPT: {result}")
        speak(result)
        try:
            if "EVENT-ALERT" in user_input:
                self.context += f"\nSystem: {user_input}\nMe: {result}"
            else:
                self.context += f"\nUser: {user_input}\nMe: {result}"
        except:
            pass