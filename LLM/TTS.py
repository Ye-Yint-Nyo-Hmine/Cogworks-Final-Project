# import the playht SDK
from pyht import Client, TTSOptions, Format
import pygame
import os
import speech_recognition as sr
import time
import sys

"""

install PyAudio, SpeechRecognition, pyht, pygame

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
        return

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
    except sr.UnknownValueError:
        print("Sorry, I did not understand the audio.")
    except sr.RequestError:
        print("Sorry, there was an issue with the speech recognition service.")




def speak(text):
    client = Client("q5HsDH2f9xajDfuzoBUcuJIBYfK2", "c413dd41ac524130ac6e783d7c9b115c")

    options = TTSOptions(
        voice="s3://voice-cloning-zero-shot/4bcdf603-fc5f-4040-a6dd-f8d0446bca9d/arthurtrainingsaad/manifest.json",
        sample_rate=44_100,
        format=Format.FORMAT_MP3,
        speed=1.1,
    )
    audio_file = "output.mp3"
    
    # Open a file to write the audio stream to
    with open(audio_file, "wb") as f:
        for chunk in client.tts(text=text, voice_engine="PlayHT2.0-turbo", options=options):
            f.write(chunk)
        f.close()
    
    # Check if the file has been created and is not empty
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
        print("Error: Audio file not created or is empty.")
        return
    
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load the mp3 file
        pygame.mixer.music.load(audio_file)

        # Play the audio file
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except pygame.error as e:
        print(f"Error during playback: {e}")

    finally:
        # Ensure that pygame mixer is stopped and quit
        if pygame.mixer.get_init():  # Check if pygame mixer is initialized
            print("stopping")
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            print("stopped")

        # Delete the audio file
        if os.path.exists(audio_file):
            print("deleted")
            os.remove(audio_file)
            print("finsihed deleted")
        pygame.quit()
        print("system stopping")
        sys.exit()

if __name__ == "__main__":
    #text = recognize_speech_from_mic()
    speak("Hello, world! The quick brown fox jumped over the lazy dog")