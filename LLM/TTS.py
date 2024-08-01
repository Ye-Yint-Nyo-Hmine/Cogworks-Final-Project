# import the playht SDK
from pyht import Client, TTSOptions, Format
import os
import speech_recognition as sr
import pyttsx3
import time

from playsound import playsound

"""

install PyAudio, SpeechRecognition, pyht, pygame pyttsx3

"""

def recognize_speech_from_mic(duration=15):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        
        start_time = time.time()
        audio = None
        
        while time.time() - start_time < duration:
            try:
                audio = recognizer.listen(source, timeout=1)
                break
            except sr.WaitTimeoutError:
                continue

    if audio is None:
        print("Recording timed out.")
        return "False audio"

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand the audio.")
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")





def speak(text, speed=None):
    """
    
    Note: this sounds a lot lamer than the pyht but the api might work tomorrow.
    """
    
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if speed is not None:
        engine.setProperty("rate", speed)
    else:
        engine.setProperty("rate", 150)
    engine.setProperty('voice', voices[0].id)
    engine.say(text)
    engine.runAndWait()

    
        


if __name__ == "__main__":

    #text = recognize_speech_from_mic()
    speak("Hello world! The quick brown fox jumped over the lazy dog")
    
    speak("I'm good, My name is Byte GPT")