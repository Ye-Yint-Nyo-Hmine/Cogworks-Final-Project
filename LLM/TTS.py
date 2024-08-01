# import the playht SDK
from pyht import Client, TTSOptions, Format
import pygame
import os
import speech_recognition as sr



"""

install PyAudio, SpeechRecognition, pyht, pygame

"""

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening ...")
        audio = r.listen(source)
        said = ""
        print("Processing ...")
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

    return said





def speak(text):
    client = Client("q5HsDH2f9xajDfuzoBUcuJIBYfK2", "c413dd41ac524130ac6e783d7c9b115c")

    options = TTSOptions(
        # this voice id can be one of our prebuilt voices or your own voice clone id, refer to the`listVoices()` method for a list of supported voices.
        voice="s3://voice-cloning-zero-shot/4bcdf603-fc5f-4040-a6dd-f8d0446bca9d/arthurtrainingsaad/manifest.json",

        # you can pass any value between 8000 and 48000, 24000 is default
        sample_rate=44_100,
    
        # the generated audio encoding, supports 'raw' | 'mp3' | 'wav' | 'ogg' | 'flac' | 'mulaw'
        format=Format.FORMAT_MP3,

        # playback rate of generated speech
        speed=1.1,
    )
    audio_file = "output.mp3"
    
    with open(audio_file, "wb") as f:
        for chunk in client.tts(text=text, voice_engine="PlayHT2.0-turbo", options=options):
            f.write(chunk)
    
    pygame.mixer.init()

    pygame.mixer.music.load(audio_file)

    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove(audio_file)

if __name__ == "__main__":
    text = get_audio()
    print(text)
    speak("Hello, world! The quick brown fox jumped over the lazy dog")