import threading
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import sounddevice as sd
import numpy as np
import wave
import pyttsx3
import requests

from config import API_KEY, REALTIME_API_URL

app = Flask(__name__)
socketio = SocketIO(app)

SAMPLERATE = 16000
DURATION = 5

# Initialize conversation history
conversation_history = []

# Global variable to control the loop
stop_requested = threading.Event()

selected_voice = None

def send_status_update(message):
    socketio.emit('status_update', message)

def send_text_update(text):
    socketio.emit('text_update', text)

def record_audio():
    send_status_update("Recording... Speak now!")
    audio_data = sd.rec(int(SAMPLERATE * DURATION), samplerate=SAMPLERATE, channels=1, dtype="int16")
    sd.wait()  # Wait until the recording is finished
    send_status_update("Recording finished.")
    return np.array(audio_data, dtype=np.int16)

def save_audio_to_wav(audio_data, filename="input.wav"):
    """Saves audio data to a WAV file."""
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(SAMPLERATE)
        wf.writeframes(audio_data.tobytes())

def transcribe_audio(audio_file_path):
    """Sends a WAV file for transcription to the OpenAI API."""
    with open(audio_file_path, "rb") as audio_file:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
        }
        files = {
            "file": audio_file,
            "model": (None, "whisper-1")  # The model for speech recognition
        }
        send_status_update("Sending audio to OpenAI API...")
        response = requests.post(f"{REALTIME_API_URL}", headers=headers, files=files)
        if response.status_code == 200:
            return response.json().get("text", "")
        else:
            send_status_update(f"Error: {response.status_code}, {response.text}")
            return None

def synthesize_speech(prompt):
    """Generates a spoken response from the OpenAI API."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # Add the new user prompt to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a friendly assistant who engages in casual conversation."}
        ] + conversation_history  # Include the conversation history
    }
    send_status_update("Generating response...")
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        response_data = response.json()
        text_response = response_data["choices"][0]["message"]["content"]
        # Add the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": text_response})
        return text_response
    else:
        send_status_update(f"Error: {response.status_code}, {response.text}")
        return None

def save_text_to_speech(text, filename="response.wav"):
    """Saves the text as a spoken response in a WAV file."""
    engine = pyttsx3.init()
    if selected_voice:
        engine.setProperty('voice', selected_voice)
    engine.save_to_file(text, filename)
    engine.runAndWait()
    return filename

def play_audio(file_path):
    """Plays a WAV file."""
    with wave.open(file_path, "rb") as wf:
        data = wf.readframes(wf.getnframes())
        sd.play(np.frombuffer(data, dtype=np.int16), samplerate=wf.getframerate())
        sd.wait()

def start_recording():
    while not stop_requested.is_set():
        audio_data = record_audio()
        save_audio_to_wav(audio_data, "input.wav")
        user_text = transcribe_audio("input.wav")

        if user_text:
            send_status_update(f"You said: {user_text}")
            send_text_update(user_text)
            if user_text.lower() in ["stop", "beenden", "ende"]:
                send_status_update("Program terminated.")
                break

            text_response = synthesize_speech(user_text)
            if text_response:
                response_audio_file = save_text_to_speech(text_response)
                send_status_update("Playing response...")
                play_audio(response_audio_file)
        else:
            send_status_update("No valid input recognized. Please try again.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global stop_requested
    stop_requested.clear()
    recording_thread = threading.Thread(target=start_recording)
    recording_thread.start()
    return "Recording started."

@app.route('/stop', methods=['POST'])
def stop():
    global stop_requested
    stop_requested.set()
    send_status_update("Program terminated.")
    return "Recording stopped."

@app.route('/voices', methods=['GET'])
def get_voices():
    """Returns a list of available voices."""
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    voice_list = [{"id": voice.id, "name": voice.name} for voice in voices]
    return jsonify(voice_list)

@app.route('/select_voice', methods=['POST'])
def select_voice():
    """Sets the selected voice for text-to-speech."""
    global selected_voice
    selected_voice = request.json.get('voice_id')
    return "Voice selected."

if __name__ == "__main__":
    socketio.run(app, debug=True)