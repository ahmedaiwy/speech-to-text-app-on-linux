import os
import sys
import shutil

def uninstall():
    # Print an uninstallation message
    print("Uninstalling Speech to Text Application...")

    # Path to the model directory who may want to be deleted
    model_dir = os.path.join(os.getcwd(), "model")

    # Attempt to remove the model directory if it exists
    if os.path.exists(model_dir):
        try:
            shutil.rmtree(model_dir)
            print(f"Removed model directory: {model_dir}")
        except Exception as e:
            print(f"Error removing model directory: {e}")

    # Optionally, you may want to remove application-specific files
    # E.g., removing desktop application shortcut if created
    shortcut_path = os.path.expanduser("~/.local/share/applications/speech_to_text.desktop")
    if os.path.exists(shortcut_path):
        try:
            os.remove(shortcut_path)
            print("Removed desktop shortcut.")
        except Exception as e:
            print(f"Error removing desktop shortcut: {e}")

    # Optionally, provide a message about removing virtualenv or environment related artifacts
    # Specifically common if you're using a virtual environment. Commented out as this is specific
    # to your environment's setup. Uncomment if it fits your uninstallation process.
    # os.system('pip uninstall -y -r requirements.txt')

    print("Uninstallation complete.")

if __name__ == "__main__":
    uninstall()