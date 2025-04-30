#!/usr/bin/env python3
"""
Voice Chat Assistant Launcher

This script launches the Voice Chat Assistant application
and ensures the Whisper model is properly located.
"""

import os
import sys
import subprocess

def main():
    """Launch the Voice Chat Assistant application."""
    print("Voice Chat Assistant Launcher")
    print("----------------------------")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for Whisper model at the specific path
    model_path = os.path.join(script_dir, "medium.pt")
    
    if os.path.exists(model_path):
        print(f"Found Whisper model at: {model_path}")
    else:
        print(f"Warning: Could not find Whisper model at {model_path}")
        print("The application will still run but may need to download the model.")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu_name}")
            print("Whisper will use GPU acceleration for faster transcription.")
        else:
            print("Warning: No GPU detected. Speech recognition will be slower.")
    except ImportError:
        print("Warning: PyTorch not installed or cannot be imported.")
    
    # Launch the main application
    main_script = os.path.join(script_dir, "main.py")
    
    if os.path.exists(main_script):
        print("\nStarting Voice Chat Assistant...\n")
        try:
            # Change to the script directory to ensure correct relative imports
            os.chdir(script_dir)
            
            # Use subprocess to run the main script
            process = subprocess.Popen([sys.executable, main_script])
            process.wait()
            
            return process.returncode
        except Exception as e:
            print(f"Error launching application: {e}")
            return 1
    else:
        print(f"Error: Could not find main.py at {main_script}")
        return 1

if __name__ == "__main__":
    sys.exit(main())