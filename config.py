"""
Configuration settings for the voice assistant application.
"""
import os

# Current directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


# API Keys - Replace with secure methods in production
GEMINI_API_KEY = "your_gemini_api_key_here"  # From chatbot.py

ELEVEN_LABS_API_KEY = "your_eleven_labs_api_key_here"




# Whisper model settings
WHISPER_MODEL_NAME = "medium"  # From whiper.py
USE_GPU_FOR_WHISPER = True     # Force GPU usage for Whisper
WHISPER_MODEL_PATH = os.path.join(CURRENT_DIR, "medium.pt")  # Direct path to model file

# Voice detection settings
ENERGY_THRESHOLD = 0.015  # From whiper.py
SILENCE_THRESHOLD = 1.0  # Seconds of silence to consider speech ended

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = "paInt16"  # For PyAudio

# Default voice ID for ElevenLabs (will be overridden by user selection)
DEFAULT_VOICE_ID = ""  # Will be selected by user in the app

# Gemini model settings
GEMINI_MODEL = "gemini-2.0-flash"
SYSTEM_INSTRUCTION = """
You are a voice-based AI assistant having a natural, conversational tone.
Always reply in a way that sounds like it's being spoken aloud by a real person.
Always respond as the same language that was given to you.
Avoid using any symbols like asterisks, emojis, or code formatting.
Avoid using enters (line spacings) in your response as lines will cause silence in the audio.
Use natural human expressions like "hmm", "let me think", "oh, got it", "well", or "sure thing" when appropriate.
Keep your responses clear, engaging, and friendly — like you're chatting with someone in person.
Keep technical explanations simple unless the user asks for more depth.
Pause words like "uh", "okay", or "so" can be used occasionally to sound more human — but don't overdo them.
"""