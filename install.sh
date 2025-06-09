#!/bin/bash

# --- Configuration ---
APP_DIR="speech_to_text_app"
VENV_DIR=".venv"
PYTHON_VERSION="python3.9" # Or python3.10, python3.11, python3.12 depending on your system
DEFAULT_VOSK_MODEL_NAME="Small English Model"
DEFAULT_VOSK_MODEL_ZIP="vosk-model-small-en-us-0.15.zip"
DEFAULT_VOSK_MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

# --- Colors for better output ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}  Speech to Text Application Installation  ${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "${YELLOW}This script will set up your application directory, install dependencies,${NC}"
echo -e "${YELLOW}and download a default Vosk model. You will need internet access.${NC}"
echo ""

# --- Check for Termux or APT ---
if command -v pkg &> /dev/null; then
    PACKAGE_MANAGER="pkg"
    echo -e "${GREEN}Detected Termux.${NC}"
elif command -v apt &> /dev/null; then
    PACKAGE_MANAGER="apt"
    echo -e "${GREEN}Detected APT (Debian/Ubuntu-based system).${NC}"
else
    echo -e "${RED}Error: Neither 'pkg' (Termux) nor 'apt' (Debian/Ubuntu) package manager found.${NC}"
    echo -e "${RED}Please ensure you are running this script on a supported environment.${NC}"
    exit 1
fi

# --- Create Application Directory ---
echo -e "${YELLOW}Creating application directory: ${APP_DIR}${NC}"
mkdir -p "$APP_DIR"
cd "$APP_DIR" || { echo -e "${RED}Failed to change to application directory. Exiting.${NC}"; exit 1; }

# --- Install System Dependencies ---
echo -e "${YELLOW}Installing system dependencies...${NC}"
if [ "$PACKAGE_MANAGER" == "pkg" ]; then
    # Termux specific packages
    pkg update -y
    pkg install -y "$PYTHON_VERSION" python-pip ffmpeg build-essential openssl-tool libffi libtool automake portaudio
elif [ "$PACKAGE_MANAGER" == "apt" ]; then
    # Debian/Ubuntu specific packages
    sudo apt update -y
    sudo apt install -y "$PYTHON_VERSION" "$PYTHON_VERSION-venv" "$PYTHON_VERSION-dev" python3-pip ffmpeg build-essential libatlas-base-dev portaudio19-dev
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install system dependencies. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}System dependencies installed.${NC}"

# --- Create and Activate Virtual Environment ---
echo -e "${YELLOW}Creating and activating Python virtual environment...${NC}"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_VERSION" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create or activate virtual environment. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment created and activated.${NC}"

# --- Install Python Dependencies ---
echo -e "${YELLOW}Installing Python packages...${NC}"
pip install --upgrade pip
pip install numpy requests vosk "pydub>=0.25.1" SpeechRecognition flask Flask-Cors "pyaudio"
# Note: "Flask-Cors" added to resolve potential CORS issues in web.
# Note: "pyaudio" is optional for web, but necessary for GUI. Included for completeness.
# Note: `pydub>=0.25.1` is specified to ensure proper ffmpeg integration.

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install Python dependencies. Exiting.${NC}"
    deactivate
    exit 1
fi
echo -e "${GREEN}Python dependencies installed.${NC}"

# --- Create Vosk Models Directory ---
echo -e "${YELLOW}Creating Vosk models directory...${NC}"
mkdir -p vosk_models
echo -e "${GREEN}Vosk models directory created.${NC}"

# --- Download Default Vosk Model ---
MODEL_PATH="vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/model"
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Downloading default Vosk model (${DEFAULT_VOSK_MODEL_NAME})...${NC}"
    mkdir -p "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}"
    wget "$DEFAULT_VOSK_MODEL_URL" -O "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/$DEFAULT_VOSK_MODEL_ZIP"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to download default Vosk model. You can try downloading it manually or selecting another model in the app.${NC}"
    else
        echo -e "${YELLOW}Extracting default Vosk model...${NC}"
        unzip "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/$DEFAULT_VOSK_MODEL_ZIP" -d "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/"
        mv "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/${DEFAULT_VOSK_MODEL_ZIP::-4}" "$MODEL_PATH" # Move extracted folder to 'model'
        rm "vosk_models/${DEFAULT_VOSK_MODEL_NAME// /_}/$DEFAULT_VOSK_MODEL_ZIP"
        echo -e "${GREEN}Default Vosk model extracted.${NC}"
    fi
else
    echo -e "${YELLOW}Default Vosk model already exists. Skipping download.${NC}"
fi


# --- Create/Update Python Backend Script (speech_to_text_backend.py) ---
echo -e "${YELLOW}Creating speech_to_text_backend.py...${NC}"
cat << 'EOF' > speech_to_text_backend.py
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
import subprocess
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # Import CORS for cross-origin requests
from werkzeug.exceptions import HTTPException
from vosk import Model, KaldiRecognizer, SetLogLevel

SetLogLevel(0)

# --- Flask Setup ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

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
GEMINI_API_KEY = "" # <<<<<<< IMPORTANT: PASTE YOUR GEMINI API KEY HERE >>>>>>>
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

    # Check if the 'model' subdirectory exists inside the extracted folder
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
                # Get the root directory name inside the zip (e.g., vosk-model-en-us-0.22)
                # and move its contents to 'model'
                first_entry = zip_ref.namelist()[0]
                zip_root_name = first_entry.split(os.sep)[0] # e.g., 'vosk-model-en-us-0.22'

                zip_ref.extractall(full_model_path)
            
            # After extraction, rename the extracted root folder to 'model'
            extracted_folder_path = os.path.join(full_model_path, zip_root_name)
            target_model_path = os.path.join(full_model_path, "model")
            
            if os.path.exists(extracted_folder_path) and not os.path.exists(target_model_path):
                os.rename(extracted_folder_path, target_model_path)
            elif os.path.exists(target_model_path):
                print(f"Warning: Target model directory already exists at {target_model_path}. Skipping rename.")

            print(f"Web: Model extracted to {full_model_path}/model.")

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
            # print(f"Web: Converted audio to WAV at {wav_temp_path} (16kHz, 1 channel, 16-bit).") # Debug log
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
                # print("Web: Online: Could not understand audio (no speech detected or ambiguous).") # Debug log
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
                    # print(f"Web: Partial recognized text (offline): {partial_text}") # Debug log
                    pass # Don't print partials to avoid spamming console

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
                if web_output_file:
                    web_output_file.close()
                    web_output_file = None
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

# --- LLM Features ---
def call_gemini_api(prompt_text):
    """Helper function to call the Gemini API."""
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set. Cannot call Gemini API.")
        return "Gemini API key is not set. Please set it in speech_to_text_backend.py."

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
        response = requests.post(api_url_with_key, headers=headers, data=json.dumps(payload), timeout=30) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        if result and result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"Gemini API response structure unexpected: {result}")
            return "Could not generate response (unexpected API structure)."
    except requests.exceptions.Timeout:
        print("Gemini API call timed out.")
        return "Error: Gemini API call timed out. Please try again."
    except requests.exceptions.ConnectionError:
        print("Gemini API connection error. Check internet connection.")
        return "Error: Could not connect to Gemini API. Check your internet connection."
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

# --- System Requirements (Common for both GUI and Web) ---
# NOTE: The GUI part and its checks are commented out for brevity in the web-only backend provided here.
# If you integrate the GUI, uncomment the relevant sections and ensure PyAudio is installed.

def check_system_requirements():
    """Checks essential system requirements for the web backend."""
    # Check for FFMPEG_PATH environment variable first, then common paths
    ffmpeg_path = os.environ.get('FFMPEG_PATH')
    if not ffmpeg_path:
        # Common Termux path for ffmpeg
        termux_ffmpeg_path = "/data/data/com.termux/files/usr/bin/ffmpeg"
        if os.path.exists(termux_ffmpeg_path):
            ffmpeg_path = termux_ffmpeg_path
        else:
            # Standard Linux paths
            if os.path.exists("/usr/local/bin/ffmpeg"):
                ffmpeg_path = "/usr/local/bin/ffmpeg"
            elif os.path.exists("/usr/bin/ffmpeg"):
                ffmpeg_path = "/usr/bin/ffmpeg"
            else:
                ffmpeg_path = "ffmpeg" # Rely on PATH if not found in common locations

    # Set pydub's converter path
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        print(f"pydub will use ffmpeg from: {AudioSegment.converter}")
    else:
        print("Warning: ffmpeg path could not be determined automatically. pydub might fail.")

    if not os.path.exists(ffmpeg_path) and ffmpeg_path != "ffmpeg":
        raise Exception(f"\nERROR: FFmpeg/libav executable not found at: {ffmpeg_path}.")
    
    try:
        # Try running ffmpeg to ensure it's executable and not corrupted
        subprocess.run([ffmpeg_path, "-version"], check=True, capture_output=True, text=True, timeout=10)
        print(f"FFmpeg/libav found and executable at: {ffmpeg_path}.")
    except subprocess.TimeoutExpired:
        raise Exception(f"\nERROR: FFmpeg/libav at {ffmpeg_path} timed out during version check. It might be stuck or corrupted.")
    except subprocess.CalledProcessError as e:
        raise Exception(f"\nERROR: FFmpeg/libav found at {ffmpeg_path} but failed to execute (return code {e.returncode}). Output: {e.stderr}")
    except FileNotFoundError:
        if ffmpeg_path == "ffmpeg": # If we're relying on PATH
             raise Exception(f"\nERROR: FFmpeg/libav not found in system PATH. Please ensure it is installed.")
        else: # If a specific path was given but not found
            raise Exception(f"\nERROR: FFmpeg/libav not found at {ffmpeg_path} (FileNotFoundError).")
    except Exception as e:
        print(f"\nWARNING: An unexpected error occurred during ffmpeg check: {e}")

# Function to run Flask app in a separate thread
def run_flask_app():
    """Runs the Flask application."""
    print("Starting Flask server...")
    try:
        check_system_requirements()
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Flask server startup error: {e}")
        # Optionally, you might want to log this to a file or notify the user in some way
        os._exit(1) # Force exit if critical error prevents server startup

# --- Main Execution Block ---
if __name__ == "__main__":
    # Start the Flask application in a separate thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    # For web-only mode, keep the main thread alive for the Flask daemon thread
    print("\nServer is running. Open your browser and navigate to http://127.0.0.1:5000")
    print("Press CTRL+C to quit the server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        # The daemon thread will automatically exit when the main thread exits.
        sys.exit(0)

EOF
echo -e "${GREEN}speech_to_text_backend.py created.${NC}"

# --- Create/Update Frontend HTML Files ---
echo -e "${YELLOW}Creating index.html, recorder_test.html, dummy_download_test.html...${NC}"

cat << 'EOF' > index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech to Text App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f4f4f4;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
        h1, h2 {
            color: #0056b3;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px 0;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .status-message {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            background-color: #e2f0d9;
            color: #28a745;
            border: 1px solid #28a745;
        }
        .error-message {
            background-color: #f8d7da;
            color: #dc3545;
            border: 1px solid #dc3545;
        }
        #transcript-area {
            width: 100%;
            height: 150px;
            border: 1px solid #ccc;
            padding: 10px;
            margin-top: 10px;
            box-sizing: border-box;
            background-color: #f9f9f9;
            border-radius: 5px;
            resize: vertical;
        }
        .mode-status {
            font-weight: bold;
            color: #28a745;
        }
        .mode-status.offline {
            color: #ffc107;
        }
        .recording-indicator {
            display: inline-block;
            width: 15px;
            height: 15px;
            background-color: red;
            border-radius: 50%;
            margin-left: 10px;
            animation: pulse 1s infinite alternate;
            display: none; /* Hidden by default */
        }
        @keyframes pulse {
            from { opacity: 1; }
            to { opacity: 0.5; }
        }
        .control-group {
            border: 1px solid #eee;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 8px;
            background-color: #fcfcfc;
        }
        .control-group h3 {
            margin-top: 0;
            color: #007bff;
        }
        #model-requirements {
            white-space: pre-wrap; /* Preserve line breaks */
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.9em;
            color: #555;
        }
        textarea {
            width: 100%;
            min-height: 100px;
            margin-top: 10px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            box-sizing: border-box;
            resize: vertical;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speech to Text Application (Web)</h1>

        <div class="control-group">
            <h3>Transcription Controls</h3>
            <p>Current Mode: <span id="mode-status" class="mode-status">Loading...</span></p>
            <p>Saving to File: <span id="save-status">Loading...</span></p>
            <button id="start-listening-btn">Start Listening</button>
            <button id="stop-listening-btn" disabled>Stop Listening</button>
            <button id="clear-transcript-btn">Clear Transcript</button>
            <span class="recording-indicator" id="recording-indicator"></span>
            <br>
            <button id="toggle-online-mode-btn">Toggle Online/Offline Mode</button>
            <button id="toggle-save-btn">Toggle Save to File</button>
        </div>

        <div class="control-group">
            <h3>Transcription Output</h3>
            <textarea id="transcript-area" placeholder="Recognized text will appear here..."></textarea>
            <button id="copy-to-clipboard-btn">Copy to Clipboard</button>
        </div>

        <div class="control-group">
            <h3>Model Management</h3>
            <label for="model-select">Select Model:</label>
            <select id="model-select">
                </select>
            <button id="download-model-btn">Download Selected Model</button>
            <button id="show-requirements-btn">Show Model Requirements</button>
            <pre id="model-requirements">Select a model and click 'Show Model Requirements'</pre>
        </div>

        <div class="control-group">
            <h3>AI Assistant (requires Gemini API Key)</h3>
            <label for="ai-input-text">Enter text for AI processing:</label>
            <textarea id="ai-input-text" placeholder="Paste recognized text here, or type your own."></textarea>
            <button id="summarize-btn">Summarize Text</button>
            <button id="action-items-btn">Extract Action Items</button>
            <label for="ai-output-text">AI Response:</label>
            <textarea id="ai-output-text" readonly placeholder="AI generated response will appear here."></textarea>
        </div>

        <p class="status-message" id="status-message"></p>
    </div>

    <script>
        const startListeningBtn = document.getElementById('start-listening-btn');
        const stopListeningBtn = document.getElementById('stop-listening-btn');
        const clearTranscriptBtn = document.getElementById('clear-transcript-btn');
        const copyToClipboardBtn = document.getElementById('copy-to-clipboard-btn');
        const toggleOnlineModeBtn = document.getElementById('toggle-online-mode-btn');
        const toggleSaveBtn = document.getElementById('toggle-save-btn');
        const downloadModelBtn = document.getElementById('download-model-btn');
        const showRequirementsBtn = document.getElementById('show-requirements-btn');
        const summarizeBtn = document.getElementById('summarize-btn');
        const actionItemsBtn = document.getElementById('action-items-btn');

        const modelSelect = document.getElementById('model-select');
        const transcriptArea = document.getElementById('transcript-area');
        const statusMessage = document.getElementById('status-message');
        const modeStatusSpan = document.getElementById('mode-status');
        const saveStatusSpan = document.getElementById('save-status');
        const recordingIndicator = document.getElementById('recording-indicator');
        const modelRequirementsPre = document.getElementById('model-requirements');
        const aiInputText = document.getElementById('ai-input-text');
        const aiOutputText = document.getElementById('ai-output-text');

        let mediaRecorder;
        let audioChunks = [];
        let isOnlineMode = false; // Initial state, will be updated by server
        let isSavingToFile = false; // Initial state, will be updated by server
        let selectedModel = "Small English Model"; // Default model

        const BACKEND_URL = 'http://127.0.0.1:5000'; // Make sure this matches your Flask server IP/port

        const MODEL_NAMES = [
            "Small English Model",
            "Medium English Model",
            "Large English Model",
            "Spanish",
            "German"
        ];

        // Populate model select dropdown
        MODEL_NAMES.forEach(modelName => {
            const option = document.createElement('option');
            option.value = modelName;
            option.textContent = modelName;
            modelSelect.appendChild(option);
        });

        // Set default selected model
        modelSelect.value = selectedModel;


        // --- UI State Management ---
        function updateUIForRecording(isRecording) {
            startListeningBtn.disabled = isRecording;
            stopListeningBtn.disabled = !isRecording;
            toggleOnlineModeBtn.disabled = isRecording;
            downloadModelBtn.disabled = isRecording;
            modelSelect.disabled = isRecording;
            showRequirementsBtn.disabled = isRecording;
            recordingIndicator.style.display = isRecording ? 'inline-block' : 'none';
            if (!isRecording) {
                // Clear any partial transcript if stopped
                transcriptArea.value = transcriptArea.value.trim();
                aiInputText.value = transcriptArea.value; // Update AI input with full transcript
            }
        }

        function displayStatus(message, isError = false) {
            statusMessage.textContent = message;
            statusMessage.className = `status-message ${isError ? 'error-message' : ''}`;
        }

        function clearStatus() {
            statusMessage.textContent = '';
            statusMessage.className = 'status-message';
        }

        async function fetchAndUpdateModeStatus() {
            try {
                const response = await fetch(`${BACKEND_URL}/get_online_mode_status`);
                const data = await response.json();
                isOnlineMode = (data.status === 'online');
                modeStatusSpan.textContent = isOnlineMode ? 'Online (Google)' : 'Offline (Vosk)';
                modeStatusSpan.className = `mode-status ${isOnlineMode ? 'online' : 'offline'}`;
                displayStatus(`Mode loaded: ${modeStatusSpan.textContent}`);
            } catch (error) {
                console.error("Error fetching online mode status:", error);
                displayStatus("Failed to load mode status.", true);
            }
        }

        async function fetchAndUpdateSaveStatus() {
            try {
                const response = await fetch(`${BACKEND_URL}/get_save_to_file_status`);
                const data = await response.json();
                isSavingToFile = (data.status === 'on');
                saveStatusSpan.textContent = isSavingToFile ? 'ON' : 'OFF';
                displayStatus(`Save to file status loaded: ${saveStatusSpan.textContent}`);
            } catch (error) {
                console.error("Error fetching save to file status:", error);
                displayStatus("Failed to load save status.", true);
            }
        }

        async function fetchAndDisplayModelRequirements(modelName) {
            try {
                const response = await fetch(`${BACKEND_URL}/get_model_requirements_web?model_name=${encodeURIComponent(modelName)}`);
                const data = await response.json();
                if (response.ok) {
                    modelRequirementsPre.textContent = data.requirements;
                } else {
                    modelRequirementsPre.textContent = `Error: ${data.error || 'Could not fetch requirements.'}`;
                }
            } catch (error) {
                console.error("Error fetching model requirements:", error);
                modelRequirementsPre.textContent = `Error: Could not fetch requirements. ${error.message}`;
            }
        }

        // --- Event Listeners ---
        startListeningBtn.addEventListener('click', async () => {
            clearStatus();
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                displayStatus('getUserMedia is not supported on your browser!', true);
                return;
            }

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    displayStatus('Recording stopped. Processing audio...');
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm;codecs=opus' });
                    audioChunks = []; // Clear chunks for next recording
                    stream.getTracks().forEach(track => track.stop()); // Stop microphone access

                    // Send the final blob to the backend
                    const formData = new FormData();
                    formData.append('audio_data', audioBlob, 'audio.webm');
                    formData.append('model_name', selectedModel);
                    formData.append('online_mode', isOnlineMode);

                    try {
                        const response = await fetch(`${BACKEND_URL}/process_audio_chunk`, {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        if (response.ok) {
                            if (data.transcript) {
                                transcriptArea.value += data.transcript + ' ';
                                transcriptArea.scrollTop = transcriptArea.scrollHeight;
                                aiInputText.value = transcriptArea.value; // Update AI input
                            } else if (data.is_silence) {
                                // Do nothing for silence, but don't show an error
                            }
                            displayStatus('Transcription successful.');
                        } else {
                            displayStatus(`Transcription error: ${data.error || 'Unknown error'}`, true);
                        }
                    } catch (error) {
                        console.error('Error sending audio to backend:', error);
                        displayStatus(`Network error during transcription: ${error.message}`, true);
                    } finally {
                        updateUIForRecording(false);
                    }
                };

                mediaRecorder.start(1000); // Start recording, collect data in 1-second chunks
                displayStatus('Recording started...');
                updateUIForRecording(true);

            } catch (err) {
                console.error('Error accessing microphone:', err);
                displayStatus(`Error accessing microphone: ${err.message}`, true);
                updateUIForRecording(false);
            }
        });

        stopListeningBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
                displayStatus('Stopping recording...');
            }
        });

        clearTranscriptBtn.addEventListener('click', () => {
            transcriptArea.value = '';
            aiInputText.value = '';
            aiOutputText.value = '';
            displayStatus('Transcript cleared.');
        });

        copyToClipboardBtn.addEventListener('click', () => {
            if (transcriptArea.value) {
                navigator.clipboard.writeText(transcriptArea.value)
                    .then(() => displayStatus('Transcript copied to clipboard!'))
                    .catch(err => displayStatus('Failed to copy text: ' + err, true));
            } else {
                displayStatus('No text to copy.', true);
            }
        });

        toggleOnlineModeBtn.addEventListener('click', async () => {
            clearStatus();
            try {
                const response = await fetch(`${BACKEND_URL}/toggle_online_mode_web`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                const data = await response.json();
                if (response.ok) {
                    isOnlineMode = (data.status === 'online');
                    modeStatusSpan.textContent = isOnlineMode ? 'Online (Google)' : 'Offline (Vosk)';
                    modeStatusSpan.className = `mode-status ${isOnlineMode ? 'online' : 'offline'}`;
                    displayStatus(data.message);
                } else {
                    displayStatus(`Error toggling mode: ${data.error || 'Unknown error'}`, true);
                }
            } catch (error) {
                console.error("Error toggling online mode:", error);
                displayStatus("Network error toggling mode.", true);
            }
        });

        toggleSaveBtn.addEventListener('click', async () => {
            clearStatus();
            try {
                const response = await fetch(`${BACKEND_URL}/toggle_save_to_file_web`, { method: 'POST', headers: { 'Content-Type': 'application/json' } });
                const data = await response.json();
                if (response.ok) {
                    isSavingToFile = (data.status === 'on');
                    saveStatusSpan.textContent = isSavingToFile ? 'ON' : 'OFF';
                    displayStatus(data.message);
                } else {
                    displayStatus(`Error toggling save: ${data.error || 'Unknown error'}`, true);
                }
            } catch (error) {
                console.error("Error toggling save to file:", error);
                displayStatus("Network error toggling save to file.", true);
            }
        });

        modelSelect.addEventListener('change', (event) => {
            selectedModel = event.target.value;
            displayStatus(`Selected model: ${selectedModel}`);
            fetchAndDisplayModelRequirements(selectedModel);
        });

        downloadModelBtn.addEventListener('click', async () => {
            clearStatus();
            displayStatus(`Downloading ${selectedModel} model... This may take a while.`);
            try {
                const response = await fetch(`${BACKEND_URL}/download_model`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: selectedModel })
                });
                const data = await response.json();
                if (response.ok) {
                    displayStatus(data.message);
                } else {
                    displayStatus(`Error downloading model: ${data.error || 'Unknown error'}`, true);
                }
            } catch (error) {
                console.error("Error downloading model:", error);
                displayStatus(`Network error during model download: ${error.message}`, true);
            }
        });

        showRequirementsBtn.addEventListener('click', () => {
            fetchAndDisplayModelRequirements(selectedModel);
        });

        summarizeBtn.addEventListener('click', async () => {
            const text = aiInputText.value.trim();
            if (!text) {
                displayStatus("Please enter text to summarize.", true);
                return;
            }
            displayStatus("Summarizing text...");
            aiOutputText.value = "Processing...";
            try {
                const response = await fetch(`${BACKEND_URL}/summarize_text`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                if (response.ok) {
                    aiOutputText.value = data.summary;
                    displayStatus("Summary generated.");
                } else {
                    aiOutputText.value = `Error: ${data.error || 'Unknown error'}`;
                    displayStatus(`Error summarizing: ${data.error || 'Unknown error'}`, true);
                }
            } catch (error) {
                console.error("Error calling summarize API:", error);
                aiOutputText.value = `Network Error: ${error.message}`;
                displayStatus(`Network error during summarization: ${error.message}`, true);
            }
        });

        actionItemsBtn.addEventListener('click', async () => {
            const text = aiInputText.value.trim();
            if (!text) {
                displayStatus("Please enter text to extract action items from.", true);
                return;
            }
            displayStatus("Extracting action items...");
            aiOutputText.value = "Processing...";
            try {
                const response = await fetch(`${BACKEND_URL}/extract_action_items`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text })
                });
                const data = await response.json();
                if (response.ok) {
                    aiOutputText.value = data.action_items;
                    displayStatus("Action items extracted.");
                } else {
                    aiOutputText.value = `Error: ${data.error || 'Unknown error'}`;
                    displayStatus(`Error extracting action items: ${data.error || 'Unknown error'}`, true);
                }
            } catch (error) {
                console.error("Error calling action items API:", error);
                aiOutputText.value = `Network Error: ${error.message}`;
                displayStatus(`Network error during action item extraction: ${error.message}`, true);
            }
        });

        // --- Initial Load ---
        window.addEventListener('load', async () => {
            updateUIForRecording(false);
            await fetchAndUpdateModeStatus();
            await fetchAndUpdateSaveStatus();
            fetchAndDisplayModelRequirements(selectedModel); // Display default model requirements
            displayStatus("Application loaded. Ready to listen.");
        });
    </script>
</body>
</html>
EOF
echo -e "${GREEN}index.html created.${NC}"

cat << 'EOF' > recorder_test.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MediaRecorder Test</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        #log {
            margin-top: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #f0f0f0;
            height: 300px;
            overflow-y: scroll;
            font-family: monospace;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>MediaRecorder Raw Test</h1>
    <p>This page tests raw audio recording using MediaRecorder and collects chunks.</p>
    <button id="startRecording">Start Recording</button>
    <button id="stopRecording" disabled>Stop Recording</button>
    <button id="clearLog">Clear Log</button>
    <div id="log"></div>

    <script>
        const startRecordingBtn = document.getElementById('startRecording');
        const stopRecordingBtn = document.getElementById('stopRecording');
        const clearLogBtn = document.getElementById('clearLog');
        const logDiv = document.getElementById('log');

        let mediaRecorder;
        let audioChunks = [];
        let chunkCount = 0;

        function log(message) {
            const p = document.createElement('p');
            p.textContent = `(index):${getLineNumber()} ${message}`;
            logDiv.appendChild(p);
            logDiv.scrollTop = logDiv.scrollHeight; // Auto-scroll to bottom
        }

        function getLineNumber() {
            try {
                throw new Error();
            } catch (e) {
                // This is a hacky way to get the line number for demonstration
                // In a real app, you might use a proper logging library or just remove line numbers
                const stack = e.stack.split('\n')[2]; // Get the line that calls this function
                const match = stack.match(/:(\d+):\d+\)$/);
                return match ? match[1] : 'N/A';
            }
        }

        startRecordingBtn.addEventListener('click', async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                log('Microphone access granted.');

                // Check supported MIME types
                const availableMimeTypes = [
                    'audio/webm;codecs=opus',
                    'audio/ogg;codecs=opus',
                    'audio/webm',
                    'audio/ogg',
                    'audio/mp4' // might require specific browser support
                ].filter(MediaRecorder.isTypeSupported);

                if (availableMimeTypes.length === 0) {
                    log('No supported audio MIME types found for MediaRecorder.', true);
                    return;
                }

                const mimeType = availableMimeTypes[0]; // Use the first supported type
                log(`MediaRecorder initialized with MIME type: ${mimeType}`);

                mediaRecorder = new MediaRecorder(stream, { mimeType: mimeType });

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                        chunkCount++;
                        log(`Audio chunk received, size: ${event.data.size}, total chunks: ${chunkCount}`);
                    }
                };

                mediaRecorder.onstart = () => {
                    log('MediaRecorder started.');
                    startRecordingBtn.disabled = true;
                    stopRecordingBtn.disabled = false;
                    chunkCount = 0;
                    audioChunks = [];
                };

                mediaRecorder.onstop = () => {
                    log('MediaRecorder stopped.');
                    log(`Total audio chunks collected: ${audioChunks.length}`);
                    startRecordingBtn.disabled = false;
                    stopRecordingBtn.disabled = true;

                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: mimeType });
                        log(`Audio blob of type: ${audioBlob.type} and size: ${audioBlob.size} bytes`);

                        // You can play back or save the blob here for further testing
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const audio = new Audio(audioUrl);
                        audio.controls = true;
                        logDiv.appendChild(audio);
                        logDiv.appendChild(document.createElement('br'));
                        log('Audio playback available below.');
                    } else {
                        log('No audio chunks collected.');
                    }
                    stream.getTracks().forEach(track => track.stop()); // Stop microphone access
                };

                // Start recording, collecting data in 1-second timeslices
                mediaRecorder.start(1000);
                log('MediaRecorder started with 1-second timeslice.');

            } catch (err) {
                log(`Error accessing microphone: ${err.name} - ${err.message}`, true);
                console.error('Error accessing microphone:', err);
                startRecordingBtn.disabled = false;
                stopRecordingBtn.disabled = true;
            }
        });

        stopRecordingBtn.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                log('mediaRecorder.stop() called.');
                mediaRecorder.stop();
            }
        });

        clearLogBtn.addEventListener('click', () => {
            logDiv.innerHTML = '';
        });
    </script>
</body>
</html>
EOF

cat << 'EOF' > dummy_download_test.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dummy Download Test</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; }
        #status { margin-top: 20px; padding: 10px; border: 1px solid #ccc; background-color: #f0f0f0; }
        #modelSelect { padding: 8px; }
    </style>
</head>
<body>
    <h1>Dummy Model Download Test</h1>
    <p>This page tests the model download endpoint on your backend.</p>

    <label for="modelSelect">Select a model to 'download':</label>
    <select id="modelSelect">
        <option value="Small English Model">Small English Model</option>
        <option value="Medium English Model">Medium English Model</option>
        <option value="Large English Model">Large English Model</option>
        <option value="Spanish">Spanish</option>
        <option value="German">German</option>
        <option value="NonExistentModel">Non-Existent Model (for error test)</option>
    </select>
    <button id="downloadButton">Start Dummy Download</button>
    <div id="status">Status: Ready</div>

    <script>
        const downloadButton = document.getElementById('downloadButton');
        const modelSelect = document.getElementById('modelSelect');
        const statusDiv = document.getElementById('status');
        const BACKEND_URL = 'http://127.0.0.1:5000'; // Make sure this matches your Flask server IP/port

        downloadButton.addEventListener('click', async () => {
            const selectedModel = modelSelect.value;
            statusDiv.textContent = `Status: Attempting to download "${selectedModel}"...`;
            statusDiv.style.color = 'orange';

            try {
                const response = await fetch(`${BACKEND_URL}/download_model`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ model_name: selectedModel })
                });

                const data = await response.json();

                if (response.ok) {
                    statusDiv.textContent = `Status: Success! ${data.message}`;
                    statusDiv.style.color = 'green';
                } else {
                    statusDiv.textContent = `Status: Error! ${data.error || 'Unknown error'}`;
                    statusDiv.style.color = 'red';
                }
            } catch (error) {
                statusDiv.textContent = `Status: Network Error! ${error.message}`;
                statusDiv.style.color = 'red';
                console.error('Error during dummy download request:', error);
            }
        });
    </script>
</body>
</html>
EOF
echo -e "${GREEN}Supplemental HTML files created.${NC}"

# --- Create Uninstall Script ---
echo -e "${YELLOW}Creating uninstall.py...${NC}"
cat << 'EOF' > uninstall.py
#!/usr/bin/env python3

import os
import shutil
import sys

def main():
    print("=======================================")
    print("  Speech to Text Application Uninstaller ")
    print("=======================================")
    print("\nThis script will attempt to remove all files created by the installer.")
    print("It will NOT remove system-wide packages like Python, pip, ffmpeg.")

    app_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Confirm with user
    confirm = input(f"\nAre you sure you want to uninstall the application from '{app_dir}'? (yes/no): ").lower()
    if confirm != 'yes':
        print("Uninstallation cancelled.")
        sys.exit(0)

    print("\nStarting uninstallation...")

    # 1. Remove Vosk models directory
    vosk_models_dir = os.path.join(app_dir, "vosk_models")
    if os.path.exists(vosk_models_dir):
        print(f"Removing Vosk models directory: {vosk_models_dir}")
        shutil.rmtree(vosk_models_dir)
    else:
        print(f"Vosk models directory not found: {vosk_models_dir}. Skipping.")

    # 2. Remove virtual environment
    venv_dir = os.path.join(app_dir, ".venv")
    if os.path.exists(venv_dir):
        print(f"Removing virtual environment: {venv_dir}")
        shutil.rmtree(venv_dir)
    else:
        print(f"Virtual environment not found: {venv_dir}. Skipping.")

    # 3. Remove application files
    files_to_remove = [
        "speech_to_text_backend.py",
        "index.html",
        "recorder_test.html",
        "dummy_download_test.html",
        "install.sh",
        "uninstall.py" # This script will remove itself at the end
    ]

    for f in files_to_remove:
        file_path = os.path.join(app_dir, f)
        if os.path.exists(file_path):
            print(f"Removing file: {file_path}")
            os.remove(file_path)
        else:
            print(f"File not found: {file_path}. Skipping.")
    
    # 4. Remove desktop shortcut (if created)
    shortcut_path = os.path.expanduser("~/.local/share/applications/speech_to_text.desktop")
    if os.path.exists(shortcut_path):
        print(f"Removing desktop shortcut: {shortcut_path}")
        os.remove(shortcut_path)
    else:
        print(f"Desktop shortcut not found: {shortcut_path}. Skipping.")

    # 5. Attempt to remove the app directory itself if empty
    try:
        # Check if the directory is empty after removing its contents
        if not os.listdir(app_dir):
            print(f"Removing empty application directory: {app_dir}")
            os.rmdir(app_dir)
        else:
            print(f"Application directory '{app_dir}' is not empty. Please remove it manually if desired.")
    except OSError as e:
        print(f"Could not remove application directory: {e}")

    print("\nUninstallation complete. Some system packages installed (like ffmpeg) were not removed and may need to be uninstalled manually if no longer needed.")

if __name__ == "__main__":
    main()
EOF
echo -e "${GREEN}uninstall.py created.${NC}"

# --- Set permissions for scripts ---
chmod +x speech_to_text_backend.py install.sh uninstall.py
echo -e "${GREEN}Permissions set for scripts.${NC}"

echo ""
echo -e "${GREEN}===========================================${NC}"
echo -e "${GREEN}  Installation Complete!                   ${NC}"
echo -e "${GREEN}===========================================${NC}"
echo ""
echo -e "${YELLOW}To run the application, navigate to the '${APP_DIR}' directory and run:${NC}"
echo -e "${YELLOW}  source ${VENV_DIR}/bin/activate${NC}"
echo -e "${YELLOW}  ${PYTHON_VERSION} speech_to_text_backend.py${NC}"
echo ""
echo -e "${YELLOW}Then, open your web browser and go to: http://127.0.0.1:5000${NC}"
echo ""
echo -e "${YELLOW}IMPORTANT: To enable AI Assistant features (Summarize, Extract Action Items),${NC}"
echo -e "${YELLOW}you need to get a Gemini API Key from Google AI Studio and paste it into:${NC}"
echo -e "${YELLOW}  'speech_to_text_backend.py' on the line 'GEMINI_API_KEY = \"\"'${NC}"
echo ""
echo -e "${YELLOW}To uninstall the application, navigate to the '${APP_DIR}' directory and run:${NC}"
echo -e "${YELLOW}  ${PYTHON_VERSION} uninstall.py${NC}"
echo ""
echo -e "${YELLOW}If you have issues, check your system's Python and FFmpeg installations.${NC}"
echo -e "${YELLOW}Enjoy!${NC}"

# Deactivate the virtual environment in the current shell after the script finishes
deactivate 2>/dev/null
