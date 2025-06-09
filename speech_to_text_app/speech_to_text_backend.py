import sys
import os
import numpy as np
import speech_recognition as sr
import requests
import json
import threading
import zipfile
import pyaudio
import time
import wave
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
import tempfile
import io
import subprocess # Kept for system requirements check

# --- UPDATED PYDUB PATH SETTING ---
# Check for FFMPEG_PATH environment variable first, then Termux default, then standard Linux default
ffmpeg_path = os.environ.get('FFMPEG_PATH')
if not ffmpeg_path:
    # Common Termux path for ffmpeg
    termux_ffmpeg_path = "/data/data/com.termux/files/usr/bin/ffmpeg"
    if os.path.exists(termux_ffmpeg_path):
        ffmpeg_path = termux_ffmpeg_path
    else:
        # Standard Linux path as a last resort (less likely for Termux)
        ffmpeg_path = "/usr/local/bin/ffmpeg"

# CORRECTED TYPO HERE: ffmpeg_path instead of ffmmp_path
AudioSegment.converter = ffmpeg_path
print(f"pydub will use ffmpeg from: {AudioSegment.converter}")
# -----------------------------------

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

# PyQt5 imports (only if running the GUI part)
PYQT_AVAILABLE = False
try:
    from PyQt5 import QtWidgets, QtCore
    from PyQt5.QtWidgets import QMessageBox
    from PyQt5.QtGui import QPixmap
    import pyperclip
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt5 or pyperclip not found. Running in web-only mode.")

from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)

# --- Flask Setup ---
app = Flask(__name__)

# Directory to store downloaded Vosk models
MODEL_BASE_DIR = os.path.join(os.getcwd(), "vosk_models")

# Global Vosk model instance for Flask app (for efficiency)
flask_vosk_models = {}

# Global state for the web version (to mimic some desktop app behaviors)
web_is_saving_to_file = False
web_output_file = None
web_last_recognized_phrase = ""
web_online_mode = False

# Model URLs for both web and desktop versions
MODEL_URLS = {
    # Updated URLs based on latest Vosk models page for better reliability
    "Large English Model": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip", # Larger graph model
    "Medium English Model": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip", # Generic US English model
    "Small English Model": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip", # Lightweight model for Android/RPi
    "Spanish": "https://alphacephei.com/vosk/models/vosk-model-es-0.42.zip",
    "German": "https://alphacephei.com/vosk/models/vosk-model-de-0.21.zip",
}

MODEL_REQUIREMENTS = {
    # Updated sizes based on the new model versions
    "Large English Model": "Size: ~1 GB\nMinimum RAM: 1 GB\nMinimum CPU: Dual-core",
    "Medium English Model": "Size: ~40 MB\nMinimum RAM: 512 MB\nMinimum CPU: Dual-core",
    "Small English Model": "Size: ~40 MB\nMinimum RAM: 256 MB\nMinimum CPU: Single-core",
    "Spanish": "Size: ~700 MB\nMinimum RAM: 512 MB\nMinimum CPU: Dual-core",
    "German": "Size: ~800 MB\nMinimum RAM: 512 MB\nMinimum CPU: Dual-core",
}

# --- Gemini API Configuration ---
GEMINI_API_KEY = "" 
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# --- General Error Handlers for Flask ---
@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Return JSON instead of HTML for HTTP errors (e.g., 404, 400)."""
    response = e.get_response()
    response.data = json.dumps({
        "error": e.name,
        "code": e.code,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Catch all unhandled exceptions and return JSON error."""
    app.logger.error(f"An unhandled error occurred: {e}", exc_info=True)
    return jsonify({"error": "An unexpected server error occurred.", "details": str(e)}), 500


# --- Flask Routes ---

@app.route("/")
def index():
    """Serves the main HTML page for the web interface."""
    return render_template("index.html")

@app.route("/recorder_test")
def recorder_test_page():
    """Serves the simple audio recorder test page."""
    return render_template("recorder_test.html")

@app.route("/dummy_download_test")
def dummy_download_test_page():
    """Serves the dummy download test page."""
    return render_template("dummy_download_test.html")

@app.route("/download_model", methods=["POST"])
def download_model_web():
    """
    Handles model download requests from the web interface.
    Downloads and extracts the selected Vosk model.
    """
    model_name = request.json.get("model_name")
    model_url = MODEL_URLS.get(model_name)

    if model_url is None:
        return jsonify({"error": "Invalid model name."}), 400

    model_dir_name = model_name.replace(" ", "_").lower()
    full_model_path = os.path.join(MODEL_BASE_DIR, model_dir_name)
    
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR)

    if os.path.exists(os.path.join(full_model_path, "model")):
        print(f"Model '{model_name}' already exists locally at {full_model_path}.")
        return jsonify({"message": f"Model '{model_name}' is already downloaded and ready."}), 200

    if not os.path.exists(full_model_path):
        os.makedirs(full_model_path)

    model_zip_path = os.path.join(full_model_path, f"{model_dir_name}.zip")

    print(f"Web: Starting download for model: {model_name} from {model_url}")

    try:
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Web: Model downloaded as {model_zip_path}.")

            print(f"Web: Extracting {model_name} model...")
            with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    zip_root_dir = member.split(os.sep)[0] + os.sep if os.sep in member else ""
                    relative_path_in_zip = os.path.relpath(member, zip_root_dir) if zip_root_dir else member
                    extracted_path = os.path.join(full_model_path, relative_path_in_zip)
                    os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                    if not member.endswith('/'):
                        with open(extracted_path, "wb") as outfile:
                            outfile.write(zip_ref.read(member))

            print(f"Web: Model extracted to {full_model_path}.")

            os.remove(model_zip_path)
            print(f"Web: Removed zip file: {model_zip_path}")

            return jsonify({"message": f"{model_name} downloaded and extracted successfully."}), 200
        else:
            print(f"Web: Failed to download model. Status code: {response.status_code}")
            return jsonify({"error": f"Failed to download model. Status code: {response.status_code}"}), 500
    except Exception as e:
        print(f"Web: Error downloading or extracting model: {e}")
        return jsonify({"error": f"Error downloading or extracting model: {str(e)}"}), 500

@app.route("/process_audio_chunk", methods=["POST"])
def process_audio_chunk_web():
    """
    Handles continuous audio input chunks from the web interface, transcribes them.
    Returns transcript and a flag indicating if speech was detected.
    """
    global web_last_recognized_phrase, web_is_saving_to_file, web_output_file, web_online_mode, flask_vosk_models

    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio data provided."}), 400

    audio_file = request.files['audio_data']
    selected_model_name = request.form.get('model_name', 'Small English Model')
    web_online_mode = request.form.get('online_mode') == 'true' 

    # Determine input format based on filename extension sent from client
    filename = audio_file.filename
    input_format = "webm" # Default
    if filename.endswith(".ogg"):
        input_format = "ogg"
    elif filename.endswith(".mp4"):
        input_format = "mp4"

    # Initialize temp file paths and objects to None outside the try block
    input_temp_path = None
    wav_temp_path = None

    transcript = ""
    is_speech_detected = False

    try:
        # Use tempfile.NamedTemporaryFile to create and manage temporary files
        # Save the incoming audio chunk to a temporary file with its original format
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{input_format}") as input_temp_file_obj:
            audio_file.save(input_temp_file_obj.name)
            input_temp_path = input_temp_file_obj.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_temp_file_obj:
            wav_temp_path = wav_temp_file_obj.name

        try:
            # pydub will now attempt to decode the specific input format
            audio = AudioSegment.from_file(input_temp_path, format=input_format)
            
            # --- CRITICAL FIX: Explicitly set sample_width and channels for Vosk compatibility ---
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2) # 16kHz, mono, 16-bit
            # ----------------------------------------------------------------------------------
            
            audio.export(wav_temp_path, format="wav")
            print(f"Web: Converted audio to WAV at {wav_temp_path} (16kHz, 1 channel, 16-bit).") # Debug log
        except CouldntDecodeError as e:
            print(f"Web: pydub CouldntDecodeError (audio conversion): {e}. Ensure ffmpeg/libav is installed and in PATH.")
            return jsonify({"error": f"Audio conversion failed: {e}. Ensure ffmpeg/libav is installed and in PATH."}), 500
        except Exception as e:
            print(f"Web: Error during pydub conversion: {e}")
            return jsonify({"error": f"Audio conversion failed: {e}"}), 500

        if web_online_mode:
            r = sr.Recognizer()
            with sr.AudioFile(wav_temp_path) as source:
                audio_data = r.record(source)
            try:
                transcript = r.recognize_google(audio_data)
                is_speech_detected = bool(transcript.strip())
                if is_speech_detected:
                    print(f"Web: Recognized text (online): {transcript}")
            except sr.UnknownValueError:
                transcript = ""
                is_speech_detected = False
                print("Web: Online: Could not understand audio (no speech detected or ambiguous).") # Debug log
            except sr.RequestError as e:
                print(f"Web: Online recognition failed for chunk: {e}")
                return jsonify({"error": f"Online recognition failed for chunk: {e}"}), 500
        else:
            model_dir_name = selected_model_name.replace(" ", "_").lower()
            vosk_model_path = os.path.join(MODEL_BASE_DIR, model_dir_name, "model")

            if vosk_model_path not in flask_vosk_models:
                if not os.path.exists(vosk_model_path):
                    return jsonify({"error": f"Vosk model for '{selected_model_name}' not found on server. Please download it first."}), 400
                try:
                    flask_vosk_models[vosk_model_path] = Model(vosk_model_path)
                    print(f"Web: Vosk Model loaded for transcription: {vosk_model_path}")
                except Exception as e:
                    return jsonify({"error": f"Failed to load Vosk model on server: {str(e)}"}), 500

            wf = wave.open(wav_temp_path, "rb")
            # Vosk expects 16kHz, 16-bit, mono. We ensure this with pydub, but a final check is good.
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                wf.close()
                print(f"Web: WAV format mismatch for Vosk: channels={wf.getnchannels()}, sampwidth={wf.getsampwidth()}, framerate={wf.getframerate()}") # Debug log
                return jsonify({"error": "Converted audio chunk must be WAV format, mono, 16-bit, 16kHz for Vosk."}), 400

            recognizer = KaldiRecognizer(flask_vosk_models[vosk_model_path], wf.getframerate())
            
            data = wf.readframes(wf.getnframes())
            if recognizer.AcceptWaveform(data):
                result_json = json.loads(recognizer.Result())
                transcript = result_json.get('text', '').strip()
                is_speech_detected = bool(transcript)
                if is_speech_detected:
                    print(f"Web: Recognized text (offline): {transcript}")
            else:
                partial_result_json = json.loads(recognizer.PartialResult())
                partial_text = partial_result_json.get('partial', '').strip()
                is_speech_detected = bool(partial_text)
                if partial_text:
                    print(f"Web: Partial recognized text (offline): {partial_text}") # Debug log

            wf.close()
        
        if is_speech_detected:
            web_last_recognized_phrase = transcript
        elif not is_speech_detected and not transcript:
             return jsonify({"transcript": "", "is_speech_detected": False, "is_silence": True}), 200

        if web_is_saving_to_file and is_speech_detected:
            try:
                if web_output_file is None or web_output_file.closed:
                    save_file_path = os.path.expanduser("~/Documents/recognized_speech_web.txt")
                    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                    web_output_file = open(save_file_path, "a", encoding="utf-8")
                    print(f"Web: Opened file for saving: {save_file_path}")

                web_output_file.write(transcript + "\n")
                web_output_file.flush()
                print(f"Web: Saved to file: {transcript}")
            except Exception as e:
                print(f"Web: Error saving to file: {e}")

    except Exception as e:
        print(f"Web: General error during transcription process for chunk: {e}")
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500
    finally:
        # Ensure temporary files are cleaned up, even if an.error occurred
        if input_temp_path and os.path.exists(input_temp_path):
            os.remove(input_temp_path)
        if wav_temp_path and os.path.exists(wav_temp_path):
            os.remove(wav_temp_path)
    
    return jsonify({"transcript": transcript, "is_speech_detected": is_speech_detected}), 200

@app.route("/toggle_save_to_file_web", methods=["POST"])
def toggle_save_to_file_web_route():
    """Toggles the web version's server-side file saving status."""
    global web_is_saving_to_file, web_output_file
    try:
        force_state = request.json.get('force_state')
        if force_state is not None:
            web_is_saving_to_file = force_state
        else:
            web_is_saving_to_file = not web_is_saving_to_file
        
        if web_is_saving_to_file:
            try:
                save_file_path = os.path.expanduser("~/Documents/recognized_speech_web.txt")
                os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                web_output_file = open(save_file_path, "a", encoding="utf-8")
                print(f"Web: Started saving recognized text to {save_file_path}")
                return jsonify({"status": "on", "message": f"Saving to {save_file_path}"}), 200
            except Exception as e:
                web_is_saving_to_file = False
                print(f"Web: Error opening file for saving: {e}")
                return jsonify({"status": "error", "message": f"Could not open file for saving: {e}"}), 500
        else:
            if web_output_file:
                web_output_file.close()
                web_output_file = None
            print("Web: Stopped saving recognized text.")
            return jsonify({"status": "off", "message": "Stopped saving to file."}), 200
    except Exception as e:
        print(f"Server error in /toggle_save_to_file_web: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/get_save_to_file_status", methods=["GET"])
def get_save_to_file_status():
    """Returns the current save-to-file status."""
    global web_is_saving_to_file
    return jsonify({"status": "on" if web_is_saving_to_file else "off"}), 200


@app.route("/toggle_online_mode_web", methods=["POST"])
def toggle_online_mode_web_route():
    """Toggles the web version's transcription mode between offline (Vosk) and online (Google SR)."""
    global web_online_mode
    try:
        force_state = request.json.get('force_state')
        if force_state is not None:
            web_online_mode = force_state
        else:
            web_online_mode = not web_online_mode

        mode_status = "online" if web_online_mode else "offline"
        print(f"Web: Switched to {mode_status} mode.")
        return jsonify({"status": mode_status, "message": f"Switched to {mode_status} mode."}), 200
    except Exception as e:
        print(f"Server error in /toggle_online_mode_web: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/get_online_mode_status", methods=["GET"])
def get_online_mode_status():
    """Returns the current online mode status."""
    global web_online_mode
    return jsonify({"status": "online" if web_online_mode else "offline"}), 200


@app.route("/get_last_phrase_web", methods=["GET"])
def get_last_phrase_web_route():
    """Returns the last recognized phrase for the web client."""
    global web_last_recognized_phrase
    return jsonify({"last_phrase": web_last_recognized_phrase}), 200

@app.route("/get_model_requirements_web", methods=["GET"])
def get_model_requirements_web_route():
    """Returns the requirements for a selected model."""
    try:
        model_name = request.args.get("model_name")
        if not model_name:
            return jsonify({"error": "Model name parameter is missing."}), 400
        requirements = MODEL_REQUIREMENTS.get(model_name, "No requirements available for this model.")
        return jsonify({"requirements": requirements}), 200
    except Exception as e:
        print(f"Server error in /get_model_requirements_web: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# --- NEW LLM FEATURES ---
def call_gemini_api(prompt_text):
    """Helper function to call the Gemini API."""
    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}]
    }
    
    api_url_with_key = GEMINI_API_URL
    if GEMINI_API_KEY:
        api_url_with_key += f"?key={GEMINI_API_KEY}"

    try:
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        
        result = response.json()
        if result and result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Gemini API response structure unexpected: {result}")
            return "Could not generate response (unexpected API structure)."
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return f"Error communicating with AI: {e}"
    except json.JSONDecodeError as e:
        print(f"Error decoding Gemini API response: {e}, Response text: {response.text}")
        return f"Error processing AI response: {e}"
    except Exception as e:
        print(f"An unexpected error occurred in Gemini API call: {e}")
        return f"An unexpected error occurred with AI: {e}"

@app.route("/summarize_text", methods=["POST"])
def summarize_text_web():
    """Summarizes the provided text using Gemini API."""
    text_to_summarize = request.json.get("text")
    if not text_to_summarize:
        return jsonify({"error": "No text provided for summarization."}), 400

    prompt = f"Please provide a concise summary of the following text:\n\n{text_to_summarize}"
    summary = call_gemini_api(prompt)
    
    return jsonify({"summary": summary}), 200

@app.route("/extract_action_items", methods=["POST"])
def extract_action_items_web():
    """Extracts action items from the provided text using Gemini API."""
    text_to_process = request.json.get("text")
    if not text_to_process:
        return jsonify({"error": "No text provided for action item extraction."}), 400

    prompt = f"From the following text, extract any clear action items or tasks. List them as bullet points. If no action items are present, state 'No action items found.'.\n\n{text_to_process}"
    action_items = call_gemini_api(prompt)
    
    return jsonify({"action_items": action_items}), 200

# --- END NEW LLM FEATURES ---

# Function to run Flask app in a separate thread
def run_flask_app():
    """Runs the Flask application."""
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# --- System Requirements (Common for both GUI and Web) ---
class SystemRequirements:
    @staticmethod
    def check_python_version():
        if sys.version_info < (3, 9):
            raise Exception("Python 3.9 or higher is required.")

    @staticmethod
    def check_dependencies():
        required_modules = ["numpy", "requests", "vosk", "zipfile", "flask", "pydub"]
        # Only add PyQt5, PyAudio, SpeechRecognition, pyperclip if PYQT_AVAILABLE is True
        if PYQT_AVAILABLE:
            required_modules.extend(["pyaudio", "speech_recognition", "PyQt5", "pyperclip"])

        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                raise Exception(f"Required Python module '{module}' is not installed. Please install it using 'pip install {module}'.")
        
        # Explicitly check ffmpeg/libav availability and executability
        global ffmpeg_path # Use the globally determined ffmpeg_path
        if not os.path.exists(ffmpeg_path):
            msg = f"\nERROR: FFmpeg/libav executable not found at the determined path: {ffmpeg_path}."
            print(msg)
            raise Exception(msg + "\nPlease ensure ffmpeg is installed and accessible (e.g., 'pkg install ffmpeg' in Termux or add to PATH).")
        
        try:
            # Try running ffmpeg to ensure it's executable and not corrupted
            subprocess.run([ffmpeg_path, "-version"], check=True, capture_output=True, text=True, timeout=10) # Added timeout
            print(f"FFmpeg/libav found and executable at: {ffmpeg_path}.")
            # Also try pydub's internal check just to be sure pydub is happy
            from pydub.utils import get_prober_name
            prober_name = get_prober_name()
            print(f"pydub recognizes {prober_name} at {AudioSegment.converter}.")
        except subprocess.TimeoutExpired:
            msg = f"\nERROR: FFmpeg/libav at {ffmpeg_path} timed out during version check. It might be stuck or corrupted."
            print(msg)
            raise Exception(msg + "\nPlease check your ffmpeg installation.")
        except subprocess.CalledProcessError as e:
            msg = f"\nERROR: FFmpeg/libav found at {ffmpeg_path} but failed to execute (return code {e.returncode}). Output: {e.stderr}"
            print(msg)
            raise Exception(msg + "\nThis might indicate a corrupted ffmpeg installation or missing system dependencies for ffmpeg.")
        except FileNotFoundError: # Should be caught by os.path.exists, but good to have
            msg = f"\nERROR: FFmpeg/libav not found at {ffmpeg_path} (FileNotFoundError)."
            print(msg)
            raise Exception(msg + "\nPlease ensure ffmpeg is installed and accessible.")
        except Exception as e:
            msg = f"\nWARNING: pydub's internal check for ffmpeg/libav failed even though ffmpeg is found and executable: {e}"
            print(msg)
            # Don't raise here, as ffmpeg itself seems fine. The pydub error might be transient.
            # The actual conversion will still use the set path.

        @staticmethod
        def check_audio_device():
            if PYQT_AVAILABLE: # Only check audio device for GUI mode
                try:
                    p = pyaudio.PyAudio()
                    device_count = p.get_device_count()
                    found_device = False
                    for i in range(device_count):
                        device_info = p.get_device_by_index(i)
                        if device_info['maxInputChannels'] > 0:
                            print(f"Audio input device found: {device_info['name']}")
                            found_device = True
                            break
                    p.terminate()
                    if not found_device:
                        print("No audio input device detected. Please connect a microphone.")
                        return False
                    return True
                except Exception as e:
                    msg = f"\nERROR: PyAudio failed to initialize or find audio devices: {e}"
                    print(msg)
                    raise Exception(msg + "\nPlease ensure 'portaudio' is installed and your microphone is connected and accessible.")
            else: # For web mode, client handles microphone access
                print("Skipping audio device check for web mode (client handles microphone).")
                return True # Assume client will handle it


    # --- PyQt5 GUI Application (Conditional) ---
    if PYQT_AVAILABLE:
        class SpeechToTextApp(QtWidgets.QWidget):
            def __init__(self):
                super().__init__()

                self.setWindowTitle("Speech to Text Application")
                self.setGeometry(100, 100, 600, 800)

                self.layout = QtWidgets.QVBoxLayout(self)
                self.adjustable_buttons = []

                self.text_area = QtWidgets.QTextEdit(self)
                self.text_area.setReadOnly(True)
                self.text_area.setStyleSheet("font-size: 14pt; background-color: #f0f0f0;")
                self.layout.addWidget(self.text_area)

                self.model_combo_box = QtWidgets.QComboBox(self)
                self.available_models = MODEL_URLS
                self.model_combo_box.addItems(self.available_models.keys())
                self.layout.addWidget(QtWidgets.QLabel("Select Speech Model:"))
                self.layout.addWidget(self.model_combo_box)

                self.show_requirements_button = self._create_button("Show Model Requirements", self.show_model_requirements)
                self.layout.addWidget(self.show_requirements_button)

                self.width_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self) # Qt5 Enum
                self.width_slider.setRange(50, 400)
                self.width_slider.setValue(150)
                self.width_slider.setToolTip("Adjust Button Width")
                self.width_slider.valueChanged.connect(self.adjust_button_width)
                self.layout.addWidget(QtWidgets.QLabel("Adjust Button Width:"))
                self.layout.addWidget(self.width_slider)

                self.height_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self) # Qt5 Enum
                self.height_slider.setRange(20, 80)
                self.height_slider.setValue(40)
                self.height_slider.setToolTip("Adjust Button Height")
                self.height_slider.valueChanged.connect(self.adjust_button_height)
                self.layout.addWidget(QtWidgets.QLabel("Adjust Button Height:"))
                self.layout.addWidget(self.height_slider)

                self.transparency_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self) # Qt5 Enum
                self.transparency_slider.setRange(0, 100)
                self.transparency_slider.setValue(100)
                self.transparency_slider.setToolTip("Adjust Transparency")
                self.transparency_slider.valueChanged.connect(self.adjust_transparency)
                self.layout.addWidget(QtWidgets.QLabel("Adjust Transparency:"))
                self.layout.addWidget(self.transparency_slider)

                self.download_button = self._create_button("Download Selected Model", self.download_model)
                self.start_button = self._create_button("Start Listening (Offline)", self.start_listening)
                self.online_button = self._create_button("Go Online", self.toggle_online_mode)
                self.stop_button = self._create_button("Stop Listening", self.stop_listening)
                self.save_to_file_button = self._create_button("Save to File: OFF", self.toggle_save_to_file)
                self.copy_to_clipboard_button = self._create_button("Copy Last Phrase to Clipboard", self.copy_last_phrase_to_clipboard)
                self.uninstall_button = self._create_button("Uninstall", self.run_uninstaller)
                self.screenshot_button = self._create_button("Take Screenshot", self.take_screenshot)
                self.about_button = self._create_button("About", self.show_about_dialog)
                self.quit_button = self._create_button("Quit", self.close)

                for button in self.adjustable_buttons:
                    self.layout.addWidget(button)

                self.setLayout(self.layout)

                # Perform system requirements check before launching the GUI
                try:
                    SystemRequirements.check_python_version()
                    SystemRequirements.check_dependencies() # This now includes PyAudio init check
                except Exception as e:
                    QMessageBox.critical(self, "System Requirements Error", str(e))
                    print(f"System requirements check failed: {e}")
                    sys.exit(1) # Exit if critical requirements are not met

                # Initialize PyAudio here, after checking system requirements
                self.p = pyaudio.PyAudio()
                self.stream = None
                self.is_listening = False
                self.online_mode = False

                self.model = None
                self.recognizer = None
                
                self.save_file_path = os.path.expanduser("~/Documents/recognized_speech.txt")
                self.is_saving_to_file = False
                self.output_file = None
                self.last_recognized_phrase = ""
                self._last_partial_len = 0

                self.model_requirements = MODEL_REQUIREMENTS

                self.adjust_button_width(self.width_slider.value())
                self.adjust_button_height(self.height_slider.value())

                self.create_desktop_shortcut()

                self.show()

            def _create_button(self, text, handler):
                button = QtWidgets.QPushButton(text, self)
                button.clicked.connect(handler)
                self.adjustable_buttons.append(button)
                return button

            def download_model(self):
                model_name = self.model_combo_box.currentText()
                model_url = self.available_models[model_name]
                model_dir_name = model_name.replace(" ", "_").lower()
                full_model_path = os.path.join(MODEL_BASE_DIR, model_dir_name)

                print(f"Desktop: Starting download for model: {model_name}")

                if not os.path.exists(full_model_path):
                    os.makedirs(full_model_path)

                if os.path.exists(os.path.join(full_model_path, "model")):
                    self.text_area.append(f"Model '{model_name}' already exists locally.")
                    print(f"Desktop: Model '{model_name}' already exists locally.")
                    return

                model_zip_path = os.path.join(full_model_path, f"{model_dir_name}.zip")

                self.text_area.append(f"Downloading {model_name} model...")
                QtWidgets.QApplication.processEvents()

                try:
                    response = requests.get(model_url, stream=True)
                    if response.status_code == 200:
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0
                        with open(model_zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                downloaded_size += len(chunk)

                        self.text_area.append(f"Model downloaded as {model_zip_path}.")
                        print(f"Desktop: Model downloaded successfully: {model_zip_path}")

                        self.text_area.append(f"Extracting {model_name} model...")
                        QtWidgets.QApplication.processEvents()
                        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                zip_root_dir = member.split(os.sep)[0] + os.sep if os.sep in member else ""
                                relative_path_in_zip = os.path.relpath(member, zip_root_dir) if zip_root_dir else member
                                extracted_path = os.path.join(full_model_path, relative_path_in_zip)
                                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                                if not member.endswith('/'):
                                    with open(extracted_path, "wb") as outfile:
                                        outfile.write(zip_ref.read(member))

                        self.text_area.append(f"Model extracted to {full_model_path}.")
                        print(f"Desktop: Model extracted to {full_model_path}")

                        os.remove(model_zip_path)
                        print(f"Desktop: Removed zip file: {model_zip_path}")

                    else:
                        self.text_area.append(f"Failed to download model. Status code: {response.status_code}")
                        print(f"Desktop: Download failed: Status code {response.status_code}")
                except Exception as e:
                    self.text_area.append(f"Error downloading or extracting model: {e}")
                    print(f"Desktop: Error during model download/extraction: {e}")

            def check_system_requirements(self):
                try:
                    SystemRequirements.check_python_version()
                    SystemRequirements.check_dependencies()
                    # Audio device check is now handled within SystemRequirements.check_dependencies
                    # and will raise an exception if PyAudio fails to initialize.
                    print("All system requirements met.")
                except Exception as e:
                    QMessageBox.critical(self, "System Requirements Error", str(e))
                    print(f"System requirements check failed: {e}")
                    sys.exit(1)

            def start_listening(self):
                if self.is_listening:
                    print("Already listening.")
                    return

                print("Starting listening process...")
                self.is_listening = True
                self.start_button.setStyleSheet("background-color: green; color: white;")

                model_name = self.model_combo_box.currentText()
                model_dir_name = model_name.replace(" ", "_").lower()
                vosk_model_path = os.path.join(MODEL_BASE_DIR, model_dir_name, "model")

                if not self.online_mode:
                    if not os.path.exists(vosk_model_path):
                        QMessageBox.warning(self, "Model Missing",
                                            f"Vosk model for '{model_name}' not found at {vosk_model_path}.\n"
                                            "Please download and extract it first.")
                        self.stop_listening()
                        return

                    try:
                        if self.model is None or (self.model and self.model.path != vosk_model_path):
                            self.model = Model(vosk_model_path)
                            self.recognizer = KaldiRecognizer(self.model, 16000)
                            print(f"Desktop: Vosk Model loaded: {vosk_model_path}")
                    except Exception as e:
                        QMessageBox.critical(self, "Vosk Model Error", f"Failed to load Vosk model: {e}")
                        print(f"Desktop: Error loading Vosk model: {e}")
                        self.stop_listening()
                        return
                try:
                    if self.stream is not None:
                        self.stream.stop_stream()
                        self.stream.close()

                    self.stream = self.p.open(format=pyaudio.paInt16,
                                                channels=1,
                                                rate=16000,
                                                input=True,
                                                frames_per_buffer=8000)
                    self.stream.start_stream()
                    print("Audio stream started.")

                    if self.online_mode:
                        self.listening_thread = QtCore.QThread()
                        self.listener_worker = OnlineListener(self.stream)
                        self.listener_worker.moveToThread(self.listening_thread)
                        self.listening_thread.started.connect(self.listener_worker.run)
                        self.listener_worker.text_recognized.connect(self.update_text_area_and_save)
                        self.listener_worker.error_occurred.connect(self.handle_online_error)
                        self.listener_worker.finished.connect(self.listening_thread.quit)
                        self.listener_worker.finished.connect(self.listener_worker.deleteLater)
                        self.listening_thread.finished.connect(self.listening_thread.deleteLater)
                        self.listening_thread.start()
                        print("Listening online...")
                    else:
                        self.listening_thread = QtCore.QThread()
                        self.listener_worker = OfflineListener(self.stream, self.recognizer)
                        self.listener_worker.moveToThread(self.listening_thread)
                        self.listening_thread.started.connect(self.listener_worker.run)
                        self.listener_worker.text_recognized.connect(self.update_text_area_and_save)
                        self.listener_worker.partial_text_recognized.connect(self.update_partial_text_area)
                        self.listener_worker.error_occurred.connect(self.handle_offline_error)
                        self.listener_worker.finished.connect(self.listening_thread.quit)
                        self.listener_worker.finished.connect(self.listener_worker.deleteLater)
                        self.listening_thread.finished.connect(self.listening_thread.deleteLater)
                        self.listening_thread.start()
                        print("Listening offline...")

                except Exception as e:
                    QMessageBox.critical(self, "Audio Stream Error", f"Error starting audio stream: {e}")
                    print(f"Error starting audio stream: {e}")
                    self.stop_listening()

            def stop_listening(self):
                if self.is_listening:
                    self.is_listening = False
                    self.start_button.setStyleSheet("background-color: red; color: white;")

                    if hasattr(self, 'listening_thread') and self.listening_thread.isRunning():
                        self.listener_worker.stop()
                        self.listening_thread.wait(2000)
                        if self.listening_thread.isRunning():
                            self.listening_thread.terminate()
                            self.listening_thread.wait(500)
                        print("Listening thread stopped.")

                    if self.stream:
                        self.stream.stop_stream()
                        self.stream.close()
                        self.stream = None
                        print("Audio stream stopped and closed.")

                    if not self.online_mode and self.recognizer:
                        final_result = self.recognizer.FinalResult()
                        if final_result:
                            try:
                                final_dict = json.loads(final_result)
                                final_text = final_dict.get('text', '').strip()
                                if final_text:
                                    self.update_text_area_and_save(final_text)
                                    print(f"Final offline result: {final_text}")
                            except json.JSONDecodeError as e:
                                print(f"Error decoding final Vosk result: {e} - Raw: {final_result}")

            @QtCore.pyqtSlot(str)
            def update_text_area_and_save(self, text):
                self.text_area.append(text)
                self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())
                self.last_recognized_phrase = text

                if self.is_saving_to_file and self.output_file:
                    try:
                        self.output_file.write(text + " ")
                        self.output_file.flush()
                        print(f"Saved to file: {text}")
                    except Exception as e:
                        QMessageBox.warning(self, "File Save Error", f"Error saving to file: {e}")
                        print(f"Error saving to file: {e}")
                        self.toggle_save_to_file()

            def toggle_save_to_file(self):
                self.is_saving_to_file = not self.is_saving_to_file
                if self.is_saving_to_file:
                    try:
                        save_file_path = os.path.expanduser("~/Documents/recognized_speech_web.txt")
                        os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
                        web_output_file = open(save_file_path, "a", encoding="utf-8")
                        print(f"Web: Started saving recognized text to {save_file_path}")
                        return jsonify({"status": "on", "message": f"Saving to {save_file_path}"}), 200
                    except Exception as e:
                        web_is_saving_to_file = False
                        print(f"Web: Error opening file for saving: {e}")
                        return jsonify({"status": "error", "message": f"Could not open file for saving: {e}"}), 500
                else:
                    if web_output_file:
                        web_output_file.close()
                        web_output_file = None
                    print("Web: Stopped saving recognized text.")
                    return jsonify({"status": "off", "message": "Stopped saving to file."}), 200

            def copy_last_phrase_to_clipboard(self):
                try:
                    import pyperclip
                    if self.last_recognized_phrase:
                        try:
                            pyperclip.copy(self.last_recognized_phrase)
                            QMessageBox.information(self, "Copied to Clipboard", "Last recognized phrase copied to clipboard!")
                            print(f"Copied to clipboard: '{self.last_recognized_phrase}'")
                        except pyperclip.PyperclipException as e:
                            QMessageBox.warning(self, "Clipboard Error", f"Could not copy to clipboard: {e}\n"
                                                                        "Please ensure you have xclip or xsel installed on Linux, "
                                                                        "or that a clipboard tool is available on your OS.")
                            print(f"Clipboard error: {e}")
                    else:
                        QMessageBox.information(self, "No Text", "No text has been recognized yet to copy.")
                        print("No text to copy to clipboard.")
                except ImportError:
                    QMessageBox.warning(self, "Missing Module", "The 'pyperclip' module is not installed. Please install it to use this feature.")
                    print("pyperclip module not found.")

            @QtCore.pyqtSlot(str)
            def update_partial_text_area(self, partial_text):
                current_text = self.text_area.toPlainText().splitlines()
                if len(current_text) > 0:
                    if self._last_partial_len > 0:
                        current_text[-1] = partial_text
                    else:
                        current_text.append(partial_text)
                    self.text_area.setPlainText('\n'.join(current_text))
                else:
                    self.text_area.setPlainText(partial_text)

                self._last_partial_len = len(partial_text)
                self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())

            @QtCore.pyqtSlot(str)
            def handle_online_error(self, error_message):
                QMessageBox.warning(self, "Online Recognition Error", error_message)
                print(f"Online recognition error: {error_message}")
                self.stop_listening()

            @QtCore.pyqtSlot(str)
            def handle_offline_error(self, error_message):
                QMessageBox.warning(self, "Offline Recognition Error", error_message)
                print(f"Offline recognition error: {error_message}")
                self.stop_listening()

            def toggle_online_mode(self):
                if self.is_listening:
                    self.stop_listening()
                    QtCore.QThread.msleep(100)

                self.online_mode = not self.online_mode
                if self.online_mode:
                    self.start_button.setText("Start Listening (Online)")
                    self.online_button.setText("Go Offline")
                    print("Switched to Online Mode.")
                else:
                    self.start_button.setText("Start Listening (Offline)")
                    self.online_button.setText("Go Online")
                    print("Switched to Offline Mode.")

            def adjust_button_width(self, value):
                for button in self.adjustable_buttons:
                    button.setMinimumWidth(value)
                    button.setMaximumWidth(value)
                    button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, button.sizePolicy().verticalPolicy()) # Qt5 Enum
                self.layout.invalidate()
                self.layout.activate()

            def adjust_button_height(self, value):
                for button in self.adjustable_buttons:
                    button.setMinimumHeight(value)
                    button.setMaximumHeight(value)
                    button.setSizePolicy(button.sizePolicy().horizontalPolicy(), QtWidgets.QSizePolicy.Fixed) # Qt5 Enum
                self.layout.invalidate()
                self.layout.activate()

            def adjust_transparency(self, value):
                alpha = value / 100.0
                self.setWindowOpacity(alpha)
                print(f"Transparency adjusted to: {alpha}")

            def create_desktop_shortcut(self):
                # Ensure the executable path is correct for the Python interpreter being used
                python_executable = sys.executable # This gets the current Python interpreter path
                script_path = os.path.abspath(__file__) # Path to the current script

                shortcut_content = f"""[Desktop Entry]
    Version=1.0
    Type=Application
    Name=Speech to Text
    Exec={python_executable} {script_path} gui
    Icon={os.path.abspath('icon.png')}
    Terminal=false
    Categories=Utility;
    """
                shortcut_path = os.path.expanduser("~/.local/share/applications/speech_to_text.desktop")

                try:
                    os.makedirs(os.path.dirname(shortcut_path), exist_ok=True)
                    with open(shortcut_path, 'w') as shortcut_file:
                        shortcut_file.write(shortcut_content.strip())
                    os.chmod(shortcut_path, 0o755)
                    print("Desktop shortcut created at ~/.local/share/applications/speech_to_text.desktop")
                except Exception as e:
                    print(f"Error creating desktop shortcut: {e}")

            def run_uninstaller(self):
                reply = QMessageBox.question(self, 'Uninstall Confirmation',
                                            "Are you sure you want to uninstall the application? This will run 'uninstall.py'.",
                                            QMessageBox.Yes | QMessageBox.No, # Qt5 Enum
                                            QMessageBox.No) # Qt5 Enum
                if reply == QMessageBox.Yes: # Qt5 Enum
                    import subprocess
                    try:
                        subprocess.Popen(['python3', os.path.join(os.path.dirname(os.path.abspath(__file__)), "uninstall.py")])
                        print("Uninstallation process initiated.")
                        self.close()
                    except Exception as e:
                        QMessageBox.critical(self, "Uninstall Error", f"Failed to start uninstaller: {e}")
                        print(f"Error starting uninstaller: {e}")
                else:
                    print("Uninstallation cancelled.")

            def closeEvent(self, event):
                self.stop_listening()
                if self.is_saving_to_file and self.output_file:
                    self.output_file.close()
                    print("Closed output file.")
                self.p.terminate()
                print("Application closed.")
                event.accept()

            def take_screenshot(self):
                pixmap = QPixmap(self.size())
                self.render(pixmap)

                default_screenshot_dir = os.path.expanduser("~/Pictures")
                if not os.path.exists(default_screenshot_dir):
                    os.makedirs(default_screenshot_dir)
                screenshot_path = os.path.join(default_screenshot_dir, "speech_to_text_screenshot.png")

                pixmap.save(screenshot_path, "PNG")
                print(f"Screenshot saved to: {screenshot_path}")
                QMessageBox.information(self, "Screenshot Taken", f"Screenshot saved to:\n{screenshot_path}")

            def show_model_requirements(self):
                model_name = self.model_combo_box.currentText()
                requirements = self.model_requirements.get(model_name, "No requirements available.")
                QMessageBox.information(self, "Model Requirements", requirements)

            def show_about_dialog(self):
                about_message = (
                    "Speech to Text Application\n\n"
                    "This application was developed with the assistance of OpenAI's language model, which provided guidance and support throughout the programming process.\n\n"
                    "We would like to express our gratitude to the dedicated team behind OpenAI and the resources available at the OpenAI platform. "
                    "Their commitment to improving artificial intelligence and enhancing human-computer interaction has has made such projects possible. "
                    "A special thanks to the developers and contributors of the Vosk speech recognition library, PyQt5, and all dependencies used in this project. "
                    "Your hard work and dedication to open-source software uphold the ideals of collaboration and innovation in the tech community. "
                    "Thank you for using our application! Stay curious and keep coding! 😊"
                )
                QMessageBox.information(self, "About", about_message)

        class OnlineListener(QtCore.QObject):
            text_recognized = QtCore.pyqtSignal(str)
            error_occurred = QtCore.pyqtSignal(str)
            finished = QtCore.pyqtSignal()

            def __init__(self, stream):
                super().__init__()
                self.stream = stream
                self.r = sr.Recognizer()
                self._running = True

            def run(self):
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    print("Closed main stream for online mode.")

                with sr.Microphone(sample_rate=16000) as source:
                    try:
                        self.r.adjust_for_ambient_noise(source)
                        print("Online mode: Adjusted for ambient noise.")
                    except Exception as e:
                        print(f"Warning: Could not adjust for ambient noise: {e}")

                    while self._running:
                        try:
                            audio = self.r.listen(source, phrase_time_limit=5)
                            text = self.r.recognize_google(audio)
                            if text:
                                self.text_recognized.emit(text)
                                print(f"Recognized text (online): {text}")
                        except sr.UnknownValueError:
                            print("Online: Could not understand audio (no speech detected or ambiguous)")
                        except sr.RequestError as e:
                            self.error_occurred.emit(f"Online: Could not request results from service; {e}")
                            print(f"Online: Could not request results from service; {e}")
                        except Exception as e:
                            self.error_occurred.emit(f"An unexpected error occurred in online listener: {e}")
                            print(f"Online: An unexpected error occurred: {e}")
                self.finished.emit()

            def stop(self):
                self._running = False

        class OfflineListener(QtCore.QObject):
            text_recognized = QtCore.pyqtSignal(str)
            partial_text_recognized = QtCore.pyqtSignal(str)
            error_occurred = QtCore.pyqtSignal(str)
            finished = QtCore.pyqtSignal()

            def __init__(self, stream, recognizer):
                super().__init__()
                self.stream = stream
                self.recognizer = recognizer
                self._running = True

            def run(self):
                print("Offline mode: Starting recognition loop.")
                while self._running:
                    try:
                        data = self.stream.read(4000, exception_on_overflow=False)

                        if not data:
                            print("No data from stream, stopping offline listener.")
                            break

                        if self.recognizer.AcceptWaveform(data):
                            result_json_str = self.recognizer.Result()
                            if result_json_str:
                                try:
                                    result_dict = json.loads(result_json_str)
                                    text = result_dict.get('text', '').strip()
                                    if text:
                                        self.text_recognized.emit(text)
                                        print(f"Recognized text (offline): {text}")
                                    self.partial_text_recognized.emit("")
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding Vosk result: {e} - Raw: {result_json_str}")
                                    self.error_occurred.emit(f"Error decoding Vosk result: {e}")
                        else:
                            partial_result_json_str = self.recognizer.PartialResult()
                            if partial_result_json_str:
                                try:
                                    partial_dict = json.loads(partial_result_json_str)
                                    partial_text = partial_dict.get('partial', '').strip()
                                    if partial_text:
                                        self.partial_text_recognized.emit(partial_text)
                                except json.JSONDecodeError as e:
                                    print(f"Error decoding partial Vosk result: {e} - Raw: {partial_result_json_str}")

                    except Exception as e:
                        self.error_occurred.emit(f"An unexpected error occurred in offline listener: {e}")
                        print(f"Offline: An unexpected error occurred: {e}")
                self.finished.emit()

            def stop(self):
                self._running = False

    # --- Main Execution Block ---
    if __name__ == "__main__":
        # Start the Flask application in a separate thread
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()

        run_gui = False
        if len(sys.argv) > 1:
            if sys.argv[1].lower() == "gui":
                run_gui = True
            elif sys.argv[1].lower() == "web":
                run_gui = False
            else:
                print(f"Unknown argument: {sys.argv[1]}. Defaulting to web-only mode.")
                print("Usage: python your_script.py [gui|web]")
                run_gui = False
        else:
            print("No mode specified. Defaulting to web-only mode.")
            print("Usage: python your_script.py [gui|web]")
        
        if run_gui:
            if PYQT_AVAILABLE:
                try:
                    # System requirements are checked inside SpeechToTextApp's __init__
                    # This ensures the QMessageBox can be used for errors.
                    app_qt = QtWidgets.QApplication(sys.argv)
                    ex = SpeechToTextApp()
                    sys.exit(app_qt.exec_())
                except Exception as e:
                    print(f"\nCRITICAL GUI STARTUP ERROR: {e}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging
                    sys.exit(1) # Exit with an error code
            else:
                print("\nGUI mode requested, but PyQt5 is not available. Please ensure PyQt5 is installed.")
                print("You can try 'pip install PyQt5' and 'pkg install python-dev xorg-xrandr xorg-xprop xorg-xwininfo' (Termux)")
                print("or 'sudo apt install python3-pyqt5' (Ubuntu) if you wish to use the GUI.")
                # Keep the Flask server running even if GUI fails to launch
                while True:
                    time.sleep(1)
        else:
            # If not running GUI, keep the main thread alive for the Flask daemon thread
            while True:
                time.sleep(1)

