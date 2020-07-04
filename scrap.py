import speech_recognition as sr
import pyttsx3
def SpeakText(command): 
    # Initialize the engine 
    engine = pyttsx3.init() 
    engine.say(command)  
    engine.runAndWait() 
r=sr.Recognizer()
with sr.Microphone() as source:
    # read the audio data from the default microphone
    try:
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio_data = r.record(source, duration=5)
        print("Recognizing...")
        # convert speech to text
        text = r.recognize_google(audio_data)
        print(text)
    except sr.UnknownValueError:
        SpeakText("Not able to detect can you please repeat?")
      
