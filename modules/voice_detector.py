"""
Voice activity detection module.
Based on detect_is_speaking.py
"""

import numpy as np
import time
import pyaudio
from threading import Thread, Event

class VoiceDetector:
    def __init__(self, energy_threshold=0.015, silence_threshold=1.0, sample_rate=16000, chunk_size=1024):
        """
        Initialize voice detector.
        
        Args:
            energy_threshold: Energy threshold to determine speech (higher = less sensitive)
            silence_threshold: Seconds of silence to consider speech ended
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
        """
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Audio processing
        self.format = pyaudio.paFloat32
        self.channels = 1
        
        # State tracking
        self.is_speaking = False
        self.prev_speaking = False
        self.speaking_start_time = 0
        self.silence_start_time = 0
        
        # For continuous monitoring
        self.stop_event = Event()
        self.audio_frames = []
        self.speech_detected_callback = None
        self.speech_ended_callback = None
        self._monitor_thread = None
        
        # PyAudio objects
        self.p = None
        self.stream = None
    
    def is_speaking_detected(self, audio_data, threshold=None):
        """
        Determine if the audio contains speech based on energy levels.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Optional override for energy threshold
            
        Returns:
            True if speaking is detected, False otherwise
        """
        if threshold is None:
            threshold = self.energy_threshold
            
        # Calculate root mean square energy
        energy = np.sqrt(np.mean(np.square(audio_data)))
        return energy > threshold
    
    def calibrate_noise(self, duration=1.0):
        """
        Calibrate noise level to adjust threshold automatically.
        
        Args:
            duration: Duration in seconds to sample background noise
        
        Returns:
            Adjusted energy threshold
        """
        if self.p is None:
            self.p = pyaudio.PyAudio()
            
        # Open stream temporarily
        stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("Calibrating noise level... Please be quiet.")
        
        # Collect noise samples
        noise_samples = []
        frames_to_collect = int(duration * self.sample_rate / self.chunk_size)
        
        for _ in range(frames_to_collect):
            audio_data = np.frombuffer(stream.read(self.chunk_size), dtype=np.float32)
            noise_samples.append(audio_data)
            time.sleep(0.01)
        
        # Clean up
        stream.stop_stream()
        stream.close()
        
        # Estimate noise floor
        noise_floor = np.mean([np.sqrt(np.mean(np.square(frame))) for frame in noise_samples])
        
        # Set threshold dynamically based on noise floor (3.5x noise floor)
        self.energy_threshold = max(self.energy_threshold, noise_floor * 3.5)
        
        print(f"Noise floor: {noise_floor:.6f}")
        print(f"Adjusted threshold: {self.energy_threshold:.6f}")
        
        return self.energy_threshold
    
    def start_monitoring(self, speech_detected_callback=None, speech_ended_callback=None):
        """
        Start monitoring audio for speech in a separate thread.
        
        Args:
            speech_detected_callback: Function to call when speech is detected, with timestamp as argument
            speech_ended_callback: Function to call when speech ends, with arguments (duration, audio_frames)
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            print("Already monitoring speech")
            return
            
        self.speech_detected_callback = speech_detected_callback
        self.speech_ended_callback = speech_ended_callback
        
        # Reset state
        self.stop_event.clear()
        self.is_speaking = False
        self.prev_speaking = False
        self.audio_frames = []
        
        # Start monitoring thread
        self._monitor_thread = Thread(target=self._monitor_speech)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        return self._monitor_thread
    
    def stop_monitoring(self):
        """Stop the speech monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self.stop_event.set()
            self._monitor_thread.join(timeout=2)
            
            # Clean up audio resources
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.p:
                self.p.terminate()
                self.p = None
    
    def _monitor_speech(self):
        """Internal function to monitor audio for speech detection."""
        # Initialize PyAudio
        if self.p is None:
            self.p = pyaudio.PyAudio()
        
        # Open stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        while not self.stop_event.is_set():
            # Read audio data
            try:
                audio_data = np.frombuffer(self.stream.read(self.chunk_size), dtype=np.float32)
            except Exception as e:
                print(f"Error reading audio: {e}")
                break
                
            # Store raw audio data (float32 format)
            raw_audio_chunk = audio_data.tobytes()
            
            # Check if audio energy indicates speaking
            raw_speaking = self.is_speaking_detected(audio_data)
            
            # Silence buffer logic
            if raw_speaking:
                # Reset silence timer when speech is detected
                self.silence_start_time = 0
                self.is_speaking = True
            elif self.is_speaking:
                # Start counting silence
                if self.silence_start_time == 0:
                    self.silence_start_time = time.time()
                # Check if silence has lasted long enough to be considered real silence
                elif time.time() - self.silence_start_time >= self.silence_threshold:
                    self.is_speaking = False
            
            # Store audio data for transcription if speaking
            if self.is_speaking:
                self.audio_frames.append(raw_audio_chunk)
            
            # State has changed
            if self.is_speaking != self.prev_speaking:
                # If started speaking
                if self.is_speaking:
                    self.speaking_start_time = time.time()
                    self.audio_frames = [raw_audio_chunk]  # Reset and add current chunk
                    
                    # Call callback if provided
                    if self.speech_detected_callback:
                        self.speech_detected_callback(self.speaking_start_time)
                        
                # If stopped speaking
                else:
                    duration = time.time() - self.speaking_start_time - self.silence_threshold
                    duration = max(0, duration)  # Ensure non-negative
                    
                    # Call callback with duration and audio frames if provided
                    if self.speech_ended_callback and self.audio_frames:
                        self.speech_ended_callback(duration, self.audio_frames.copy())
                        
                    # Clear audio frames
                    self.audio_frames = []
                    
            self.prev_speaking = self.is_speaking
            
            # Small delay to prevent CPU overuse
            time.sleep(0.01)
            
        # Clean up (if not already done)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

# Example usage (if running this file directly)
if __name__ == "__main__":
    detector = VoiceDetector()
    
    # Optional: Calibrate noise level
    detector.calibrate_noise()
    
    # Define callbacks for demonstration
    def on_speech_detected(timestamp):
        print(f"Speech detected at {timestamp}")
        
    def on_speech_ended(duration, frames):
        print(f"Speech ended after {duration:.2f} seconds ({len(frames)} frames)")
    
    # Start monitoring with callbacks
    detector.start_monitoring(
        speech_detected_callback=on_speech_detected,
        speech_ended_callback=on_speech_ended
    )
    
    # Run for a while then stop
    try:
        print("Monitoring speech... Press Ctrl+C to stop")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
        detector.stop_monitoring()