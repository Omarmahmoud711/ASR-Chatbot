"""
Speech-to-text module using Whisper.
Based on whiper.py
"""

import os
import tempfile
import wave
import numpy as np
import whisper
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

class SpeechToText:
    def __init__(self, model_name="medium", model_path=None):
        """
        Initialize speech-to-text module.
        
        Args:
            model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
            model_path: Direct path to the model file (.pt)
        """
        self.model_name = model_name
        self.model_path = model_path
        
        # Force GPU usage
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.device = "cuda"
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("WARNING: GPU not detected, falling back to CPU")
        
        # Load the model
        if model_path:
            print(f"Loading Whisper model directly from: {model_path}")
        else:
            print(f"Loading Whisper {model_name} model...")
            
        self.model = self._load_model()
        
    def _load_model(self):
        """Load and return the Whisper model."""
        try:
            # Check if direct model path is provided
            if self.model_path and os.path.exists(self.model_path):
                # Load model directly from specified path
                print(f"Loading model from specific path: {self.model_path}")
                
                # Use the provided path's directory for the download_root
                model = whisper.load_model(
                    name=self.model_name,  
                    device=self.device,
                    download_root=os.path.dirname(self.model_path)
                )
                
                print(f"Whisper model loaded successfully from specific path! Using {self.device.upper()}")
                
                # Optimize model for inference
                if self.use_gpu:
                    model.to(self.device)
                    # Enable CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                
                return model
                
            else:
                # If direct path not provided or invalid, fall back to standard loading
                print(f"Direct model path not provided or invalid. Using standard loading method.")
                model = whisper.load_model(
                    self.model_name,
                    device=self.device
                )
                
                # Optimize model for inference
                if self.use_gpu:
                    model.to(self.device)
                    # Enable CUDA optimizations
                    torch.backends.cudnn.benchmark = True
                    
                print(f"Whisper model loaded using standard method! Using {self.device.upper()}")
                return model
                
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Traceback:", torch.cuda.memory_summary(device=self.device) if self.use_gpu else "N/A")
            raise
            
    def save_audio_to_temp(self, audio_frames, sample_rate=16000, channels=1):
        """
        Save audio frames to a temporary WAV file and return the filename.
        
        Args:
            audio_frames: List of audio frame bytes
            sample_rate: Audio sample rate
            channels: Number of audio channels
            
        Returns:
            Path to temporary audio file
        """
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        # Save the audio frames to the file
        with wave.open(temp_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 2 bytes for 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(audio_frames))
        
        return temp_filename
    
    def transcribe_audio_file(self, audio_file):
        """
        Transcribe audio from a file using Whisper.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Tuple of (transcription_text, detected_language)
        """
        try:
            # Read audio data as NumPy array
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Warning if sample rate doesn't match what Whisper expects
            if rate != 16000:
                print(f"Warning: Whisper expects 16kHz audio but got {rate}Hz")
            
            # Utilize GPU for faster transcription
            with torch.cuda.amp.autocast(enabled=self.use_gpu):
                # Enable GPU optimization flags if available
                if self.use_gpu:
                    torch.backends.cudnn.benchmark = True
                    # Use half-precision for faster processing
                    fp16_setting = True
                else:
                    fp16_setting = False
                
                # Transcribe using Whisper with optimized settings
                result = self.model.transcribe(
                    audio_np, 
                    fp16=fp16_setting,
                    temperature=0.0,  # Use greedy decoding for speed
                    beam_size=5 if self.use_gpu else 1  # Beam search on GPU for better accuracy
                )
            
            # Get transcript and language
            transcript = result.get("text", "").strip()
            language = result.get("language", "unknown")
            
            return transcript, language
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return "Transcription failed", "unknown"
        finally:
            # Clean up the file if it exists
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception:
                pass
                
    def transcribe_audio_frames(self, audio_frames, sample_rate=16000, channels=1):
        """
        Transcribe audio from frames.
        
        Args:
            audio_frames: List of audio frame bytes
            sample_rate: Audio sample rate
            channels: Number of audio channels
            
        Returns:
            Tuple of (transcription_text, detected_language)
        """
        if not audio_frames or len(audio_frames) < 10:
            return "Audio too short", "unknown"
            
        # Save to temp file
        temp_audio_file = self.save_audio_to_temp(audio_frames, sample_rate, channels)
        
        # Transcribe
        return self.transcribe_audio_file(temp_audio_file)
        
# Example usage
if __name__ == "__main__":
    # This is just a basic test to ensure the module loads correctly
    from config import WHISPER_MODEL_PATH
    transcriber = SpeechToText(model_name="medium", model_path=WHISPER_MODEL_PATH)
    print("Speech-to-Text module initialized successfully")