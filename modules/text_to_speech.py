"""
Text-to-speech module using ElevenLabs.
Based on eleven.py
"""

import tempfile
import os
import pygame
from elevenlabs.client import ElevenLabs
import time
import threading

class TextToSpeech:
    def __init__(self, api_key):
        """
        Initialize text-to-speech module.
        
        Args:
            api_key: ElevenLabs API key
        """
        # Initialize ElevenLabs client
        self.client = ElevenLabs(api_key=api_key)
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Store available voices
        self.voices = self.get_voices()
        
        # Currently selected voice
        self.current_voice_id = None
        
    def get_voices(self):
        """
        Fetch and return available voices.
        
        Returns:
            List of voice objects
        """
        try:
            voices_response = self.client.voices.get_all()
            voices = voices_response.voices if hasattr(voices_response, 'voices') else voices_response
            return voices
        except Exception as e:
            print(f"Error fetching voices: {e}")
            return []
    
    def get_voice_names(self):
        """
        Get a list of voice names and IDs for display.
        
        Returns:
            List of (name, voice_id) tuples
        """
        return [(voice.name, voice.voice_id) for voice in self.voices]
            
    def set_voice(self, voice_id):
        """
        Set the current voice.
        
        Args:
            voice_id: ElevenLabs voice ID
            
        Returns:
            True if successful, False otherwise
        """
        # Validate voice ID
        valid_ids = [voice.voice_id for voice in self.voices]
        if voice_id not in valid_ids:
            print(f"Invalid voice ID: {voice_id}")
            return False
            
        self.current_voice_id = voice_id
        return True
    
    def speak(self, text, voice_id=None, model_id="eleven_multilingual_v2"):
        """
        Convert text to speech and play audio.
        
        Args:
            text: Text to convert to speech
            voice_id: Voice ID to use (defaults to current voice)
            model_id: ElevenLabs model ID
            
        Returns:
            True if successful, False otherwise
        """
        if not text:
            print("No text provided")
            return False
            
        # Use provided voice_id or current voice
        voice_id = voice_id or self.current_voice_id
        if not voice_id:
            print("No voice selected")
            return False
            
        try:
            # Convert text to speech as a stream
            audio_stream = self.client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=voice_id,
                model_id=model_id
            )
            
            # Use a temporary file to store audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                output_file = temp_audio.name
                # Collect and write all bytes from the stream
                for chunk in audio_stream:
                    if isinstance(chunk, bytes):
                        temp_audio.write(chunk)
            
            # Play the audio using pygame
            try:
                pygame.mixer.music.load(output_file)
                pygame.mixer.music.play()
                
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Ensure a small pause between audio segments
                pygame.time.wait(100)
                
                return True
                
            except Exception as play_error:
                print(f"Could not play audio: {play_error}")
                return False
                
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(output_file)
                except Exception:
                    pass
                    
        except Exception as e:
            print(f"Text-to-speech error: {e}")
            return False
            
    def is_speaking(self):
        """
        Check if audio is currently playing.
        
        Returns:
            True if speaking, False otherwise
        """
        return pygame.mixer.music.get_busy()
    
    def stop_speaking(self):
        """Stop any current playback."""
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()

    def speak_with_callback(self, text, callback=None, voice_id=None, model_id="eleven_multilingual_v2"):
        """
        Convert text to speech and play audio with callback when complete.
        
        Args:
            text: Text to convert to speech
            callback: Function to call when playback completes
            voice_id: Voice ID to use (defaults to current voice)
            model_id: ElevenLabs model ID
            
        Returns:
            True if successful, False otherwise
        """
        result = self.speak(text, voice_id, model_id)
        
        # Call the callback function when playback finishes in a separate thread
        if callback and result:
            # Create a thread to monitor playback and trigger callback
            def monitor_playback():
                # Wait for playback to finish
                while self.is_speaking():
                    time.sleep(0.1)
                
                # Call the callback function
                callback()
            
            threading.Thread(target=monitor_playback, daemon=True).start()
        
        return result

# Example usage
if __name__ == "__main__":
    from config import ELEVEN_LABS_API_KEY
    
    tts = TextToSpeech(api_key=ELEVEN_LABS_API_KEY)
    
    # Display available voices
    voices = tts.get_voice_names()
    print("Available voices:")
    for i, (name, voice_id) in enumerate(voices, 1):
        print(f"{i}. {name} ({voice_id})")
        
    # Select a voice for testing
    if voices:
        voice_name, voice_id = voices[0]
        tts.set_voice(voice_id)
        
        # Test speech
        print(f"Testing speech with voice: {voice_name}")
        tts.speak("Hello, this is a test of the ElevenLabs text to speech system.")
    else:
        print("No voices available")