"""
Common audio utilities for the voice assistant.
"""

import numpy as np
import pyaudio
import wave
import tempfile
import os
from typing import List, Tuple

# Common audio parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHANNELS = 1

def convert_audio_format(audio_frames: List[bytes], 
                         from_format: str, 
                         to_format: str) -> List[bytes]:
    """
    Convert audio frames between formats (e.g., paFloat32 to paInt16).
    
    Args:
        audio_frames: List of audio frame bytes
        from_format: Source PyAudio format (e.g., 'paFloat32')
        to_format: Target PyAudio format (e.g., 'paInt16')
        
    Returns:
        List of converted audio frames
    """
    # Map format strings to numpy dtypes and value ranges
    format_map = {
        'paFloat32': (np.float32, -1.0, 1.0),
        'paInt16': (np.int16, -32768, 32767),
        'paInt32': (np.int32, -2147483648, 2147483647),
        'paInt8': (np.int8, -128, 127),
        'paUInt8': (np.uint8, 0, 255)
    }
    
    if from_format not in format_map or to_format not in format_map:
        raise ValueError(f"Unsupported format conversion: {from_format} to {to_format}")
        
    from_dtype, from_min, from_max = format_map[from_format]
    to_dtype, to_min, to_max = format_map[to_format]
    
    # Convert frames
    converted_frames = []
    for frame in audio_frames:
        # Convert to numpy array
        np_audio = np.frombuffer(frame, dtype=from_dtype)
        
        # Normalize to [-1.0, 1.0] range
        if from_format != 'paFloat32':
            np_audio = np_audio.astype(np.float32) / (from_max if from_max > 0 else -from_min)
        
        # Convert to target range
        if to_format != 'paFloat32':
            np_audio = (np_audio * (to_max if to_max > 0 else -to_min)).astype(to_dtype)
        
        # Convert back to bytes
        converted_frames.append(np_audio.tobytes())
    
    return converted_frames

def save_audio_to_wav(audio_frames: List[bytes], 
                      filename: str,
                      sample_rate: int = DEFAULT_SAMPLE_RATE,
                      channels: int = DEFAULT_CHANNELS,
                      sample_width: int = 2) -> str:
    """
    Save audio frames to a WAV file.
    
    Args:
        audio_frames: List of audio frame bytes
        filename: Output filename (will append .wav if not present)
        sample_rate: Audio sample rate
        channels: Number of audio channels
        sample_width: Sample width in bytes (2 for 16-bit)
        
    Returns:
        Path to saved file
    """
    # Ensure filename has .wav extension
    if not filename.lower().endswith('.wav'):
        filename += '.wav'
        
    # Save the audio frames to the file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(audio_frames))
    
    return filename

def save_audio_to_temp(audio_frames: List[bytes],
                       sample_rate: int = DEFAULT_SAMPLE_RATE,
                       channels: int = DEFAULT_CHANNELS,
                       sample_width: int = 2) -> str:
    """
    Save audio frames to a temporary WAV file.
    
    Args:
        audio_frames: List of audio frame bytes
        sample_rate: Audio sample rate
        channels: Number of audio channels
        sample_width: Sample width in bytes (2 for 16-bit)
        
    Returns:
        Path to temporary file
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_filename = temp_file.name
    
    # Save the audio frames to the file
    save_audio_to_wav(
        audio_frames=audio_frames,
        filename=temp_filename,
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width
    )
    
    return temp_filename

def convert_between_pyaudio_formats(format_str: str) -> int:
    """
    Convert format string to PyAudio format constant.
    
    Args:
        format_str: Format string (e.g., 'paFloat32')
        
    Returns:
        PyAudio format constant
    """
    format_map = {
        'paFloat32': pyaudio.paFloat32,
        'paInt16': pyaudio.paInt16,
        'paInt32': pyaudio.paInt32,
        'paInt8': pyaudio.paInt8,
        'paUInt8': pyaudio.paUInt8
    }
    
    if format_str in format_map:
        return format_map[format_str]
    else:
        raise ValueError(f"Unsupported PyAudio format: {format_str}")

def get_pyaudio_format_sample_width(format_constant: int) -> int:
    """
    Get sample width in bytes for a PyAudio format constant.
    
    Args:
        format_constant: PyAudio format constant
        
    Returns:
        Sample width in bytes
    """
    format_widths = {
        pyaudio.paFloat32: 4,
        pyaudio.paInt16: 2,
        pyaudio.paInt32: 4,
        pyaudio.paInt8: 1,
        pyaudio.paUInt8: 1
    }
    
    if format_constant in format_widths:
        return format_widths[format_constant]
    else:
        raise ValueError(f"Unsupported PyAudio format constant: {format_constant}")