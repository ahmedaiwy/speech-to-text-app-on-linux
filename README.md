Setting up a new application can feel daunting, but don't worry! This guide is designed to walk you through getting your Speech to Text application up and running, even if you're new to this.

Welcome to Your Speech to Text App!
This application lets you convert spoken words into text using your computer's microphone. It can work in two ways:

Offline: Using a pre-downloaded model (Vosk) directly on your device.
Online: Using Google's powerful speech recognition service (requires internet).
It also includes some cool AI features like summarizing text and extracting action items, powered by Google's Gemini AI.

What You'll Find Here
Once you've run the installation script, you'll have these important files in your speech_to_text_app folder:

install.sh: The script you just ran to set everything up.
speech_to_text_backend.py: This is the "brain" of your app. It's a Python program that handles the speech recognition and AI tasks.
index.html: This is the "face" of your app. It's a web page you'll open in your browser to interact with the app,and  or a beautiful gui interface using pyqt5"sudo python3.x speech_to_text_backend.py gui
for web sudo python3.x speech_to_text_backend.py web".
vosk_models/: A folder where your downloaded offline speech recognition models will be stored.
.venv/: A special folder that holds all the Python tools your app needs, keeping them separate from other Python projects on your system.
uninstall.py: A helpful script to remove the entire application if you ever need to.
First Time Setup (If You Haven't Already!)
If you haven't run the install.sh script yet, here's how to do it. This only needs to be done once.

Open your Terminal/Command Line:
On Android (Termux): Open the Termux app.
On Linux: Open your terminal application (usually found in your applications menu).
Navigate to where you saved install.sh:
If you saved it directly in your home folder, you might already be there.
If not, use the cd command. For example, if it's in a "Downloads" folder: cd Downloads
Make the script executable:
Bash

chmod +x install.sh
Run the installation script:
Bash

./install.sh
The script will print messages as it works, letting you know what it's doing (installing tools, downloading files, etc.). This might take a few minutes depending on your internet speed and device.
How to Run Your App
Once the installation is complete (or if you've already installed it):

Open your Terminal/Command Line (Termux or Linux terminal).
Go into your application's folder: The installer created a folder called speech_to_text_app wherever you ran install.sh.
Bash

cd speech_to_text_app
Activate the app's special environment: This step ensures your app uses its own set of Python tools.
Bash

source .venv/bin/activate
You'll notice (.venv) appear at the beginning of your command line prompt, indicating the environment is active.
Start the application's "brain" (the backend server):
Bash

python3.9 speech_to_text_backend.py
Heads up: If python3.9 gives an error, try just python3 or python or python3.10, depending on what your system uses. The script will usually tell you what version it found.
You'll see messages like "Starting Flask server..." and "Running on http://127.0.0.1:5000". This means your app is running!
Open the web interface in your browser: Open your favorite web browser (like Chrome, Firefox, or your phone's default browser) and go to this address:
http://127.0.0.1:5000
You should see the "Speech to Text Application (Web)" page!
Using the App & Pro Tips
Start Listening: Click "Start Listening" to begin transcribing your speech. Click "Stop Listening" when done.
Online vs. Offline Mode:
By default, it uses Offline (Vosk) mode. You'll need to download a Vosk model first from the "Model Management" section (e.g., "Small English Model" is a good start).
For generally better accuracy (requires internet), click "Toggle Online/Offline Mode". It will switch to Google's speech recognition.
AI Assistant:
To use "Summarize Text" and "Extract Action Items", you'll need an API key from Google's Gemini AI.
How to get an API Key:
Go to Google AI Studio.
Generate a new API key.
Copy the key.
In your terminal, press CTRL+C to stop the running app.
Open the speech_to_text_backend.py file in a text editor (e.g., nano speech_to_text_backend.py).
Find the line that looks like: GEMINI_API_KEY = ""
Paste your key between the quotes: GEMINI_API_KEY = "YOUR_PASTED_API_KEY_HERE"
Save the file and restart the app (repeat steps 3 and 4 from "How to Run Your App").
Saving to File: Click "Toggle Save to File" to save your recognized text to a file named recognized_speech_web.txt in your Documents folder.
Clear Transcript: Clears the text box.
Copy to Clipboard: Copies the text from the transcript area.
When You're Done (Stopping the App)
To stop the backend server:

Go back to your Terminal/Command Line where the speech_to_text_backend.py is running.
Press CTRL + C (hold Control and press the C key).
You'll see "Shutting down server..." and your command prompt will return.
To leave the virtual environment, type: deactivate
Need to Uninstall?
If you ever want to remove the application and all its files:

Open your Terminal/Command Line.
Go into your application's folder: cd speech_to_text_app
Run the uninstall script:
Bash

python3 uninstall.py
Follow the prompts.
That's it! You're ready to start transcribing speech. If you encounter any issues, don't hesitate to ask for help!






