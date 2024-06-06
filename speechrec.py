import speech_recognition as sr
import pyttsx3
import langid
from app import getApiResponse
from gtts import gTTS
import os

recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Get a list of available voices
voices = engine.getProperty('voices')



# Function to test voices
def test_voices():
    for voice in voices:
        engine.setProperty('voice', voice.id)
        print(f"Testing voice: {voice.name} ({voice.id})")
        engine.say("Hello, this is a sample text to test if the voice is female or male.")
        engine.runAndWait()

# Call the test_voices function to test the voices
test_voices()





# Set the desired voice for English and Turkish
english_voice = voices[1].id

# Set properties 
engine.setProperty('rate', 150)  # Speed of speech

#speak: Function to convert text to speech.
def speak(text, language='en'):
    if language == 'tr':
        tts = gTTS(text=text, lang='tr')
        tts.save("temp.mp3")
        os.system("start temp.mp3")  # Adjust the command based on your system
    else:
        engine.setProperty('voice', english_voice)  # Change voice for English
        #pyttsx3
        engine.say(text)
        engine.runAndWait()


while True:
    try:
        with sr.Microphone() as mic:
            print("Listening...")
            audio = recognizer.listen(mic)
            print("Recognizing...")

            try:
                # Detect language using langid library
                lang, confidence = langid.classify(recognizer.recognize_google(audio, language='tr'))

                # Set the language for speech recognition
                if lang == 'tr':
                    text = recognizer.recognize_google(audio, language='tr-TR')
                    lang_for_speech = 'tr'  # Set language for speech synthesis
                else:
                    text = recognizer.recognize_google(audio, language='en-US')
                    lang_for_speech = 'en'  # Set language for speech synthesis

                print("Recognized:", text)
                ###############################
                #start speakin(Assisant)

                speak("Getting the Response", language=lang_for_speech)

                response = getApiResponse(text)
                print(response)

                speak(response, language=lang_for_speech)

            except sr.UnknownValueError:
                print("Could not understand audio.")
                speak("Could not understand audio.", language='en')
            except sr.RequestError as e:
                print(f"Speech recognition request failed: {e}")
                speak("Speech recognition request failed.", language='en')
            except Exception as e:
                print(f"Unexpected error: {e}")
                speak("An error occurred.", language='en')

    except KeyboardInterrupt:
        print("Program interrupted by user")
        break
    except Exception as e:
        print(f"Unexpected error: {e}")
        break
