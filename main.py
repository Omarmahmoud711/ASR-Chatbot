"""
Voice Chat Assistant - Modern GUI Application

This application integrates speech recognition, AI chatbot, and
text-to-speech capabilities with a modern GUI interface.
"""

import os
import time
import threading
import customtkinter as ctk
from customtkinter import ThemeManager
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pyaudio
import json

# Import app modules
from modules.voice_detector import VoiceDetector
from modules.speech_to_text import SpeechToText
from modules.text_to_speech import TextToSpeech
from modules.ai_chatbot import AIChatbot

# Import config
import config

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

# Constants
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

class ScrollableChatFrame(ctk.CTkScrollableFrame):
    """Custom scrollable frame for chat messages with message bubbles."""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        
        # Message container (to allow dynamic resizing)
        self.message_widgets = []
        self.current_row = 0
        
        # Store user and assistant colors
        self.user_bubble_color = "#DCF8C6"  # Light green for user (like WhatsApp)
        self.assistant_bubble_color = "#E6E6E6"  # Light gray for assistant
        self.system_bubble_color = "#F2F2F2"  # Very light gray for system
        
        # Font configurations
        self.normal_font = ("Segoe UI", 11)
        self.bold_font = ("Segoe UI", 11, "bold")
        self.small_font = ("Segoe UI", 9)
        
        # Create containers dictionary
        self.containers = {}
    
    def add_message(self, sender, message, timestamp=None):
        """Add a message to the chat."""
        if not timestamp:
            timestamp = time.strftime("%H:%M")
            
        # Create container frame for the message
        if sender == "You":
            container = ctk.CTkFrame(self, corner_radius=10, fg_color=self.user_bubble_color)
            container.grid(row=self.current_row, column=0, sticky="e", padx=(60, 10), pady=5)
            align = "right"
            text_color = "#000000"  # Black text on light background
        elif sender == "Assistant":
            container = ctk.CTkFrame(self, corner_radius=10, fg_color=self.assistant_bubble_color)
            container.grid(row=self.current_row, column=0, sticky="w", padx=(10, 60), pady=5)
            align = "left"
            text_color = "#000000"  # Black text on light background
        else:  # System
            container = ctk.CTkFrame(self, corner_radius=10, fg_color=self.system_bubble_color)
            container.grid(row=self.current_row, column=0, sticky="", padx=10, pady=5)
            align = "center"
            text_color = "#666666"  # Dark gray for system messages
        
        # Configure container
        container.grid_columnconfigure(0, weight=1)
        
        # Add sender label
        sender_label = ctk.CTkLabel(
            container, 
            text=sender, 
            font=self.bold_font,
            text_color=text_color,
            anchor="w",
            justify="left"
        )
        sender_label.grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
        
        # Add message text with word wrapping
        message_label = ctk.CTkLabel(
            container, 
            text=message,
            font=self.normal_font,
            text_color=text_color,
            anchor="w",
            justify="left",
            wraplength=400  # Set wraplength for text wrapping
        )
        message_label.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 0))
        
        # Add timestamp
        time_label = ctk.CTkLabel(
            container, 
            text=timestamp,
            font=self.small_font,
            text_color="#888888",
            anchor="e"
        )
        time_label.grid(row=2, column=0, sticky="e", padx=10, pady=(0, 5))
        
        # Store message widget reference
        self.message_widgets.append((container, sender_label, message_label, time_label))
        
        # Save container to dictionary
        self.containers[self.current_row] = container
        
        # Increment row counter
        self.current_row += 1
        
        # Scroll to the latest message
        self.after(100, self._scroll_to_bottom)
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the chat."""
        self._parent_canvas.yview_moveto(1.0)
    
    def clear(self):
        """Clear all messages from the chat."""
        for container, *_ in self.message_widgets:
            container.destroy()
        
        self.message_widgets = []
        self.current_row = 0

class ModernVoiceAssistantApp(ctk.CTk):
    def __init__(self):
        """Initialize the Voice Assistant application."""
        super().__init__()
        
        # Configure window
        self.title("Voice Chat Assistant")
        self.geometry("900x700")
        self.minsize(700, 500)
        
        # Set icon
        # self.iconbitmap(os.path.join(ASSETS_DIR, "app_icon.ico"))  # Uncomment when icon is available
        
        # Load assets
        self.load_assets()
        
        # Set up the UI
        self.setup_ui()
        
        # Initialize components
        self.initialize_components()
        
        # State variables
        self.is_listening = False
        self.is_speaking = False
        self.processing_speech = False
        self.animation_after_id = None
        
        # Pulse animation counter
        self.pulse_counter = 0
        self.pulse_direction = 1
        
        # Protocol handler for closing the window
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_assets(self):
        """Load images and icons for the application."""
        # Ensure assets directory exists
        os.makedirs(ASSETS_DIR, exist_ok=True)
        
        # Define default icons if assets don't exist
        self.default_size = (24, 24)
        
        # Check if assets directory has images, otherwise create placeholders
        self.icons = {
            "mic": self.load_or_create_icon("mic.png", "🎤"),
            "mic_active": self.load_or_create_icon("mic_active.png", "🎤", color="#ff0000"),
            "send": self.load_or_create_icon("send.png", "➤"),
            "settings": self.load_or_create_icon("settings.png", "⚙️"),
            "theme": self.load_or_create_icon("theme.png", "🐸"),
            "reset": self.load_or_create_icon("reset.png", "🔄"),
            "user": self.load_or_create_icon("user.png", "👤"),
            "assistant": self.load_or_create_icon("assistant.png", "🤖")
        }
        
    def load_or_create_icon(self, filename, text_fallback, size=None, color=None):
        """Load an icon or create a text-based button if the icon doesn't exist."""
        if size is None:
            size = self.default_size
            
        filepath = os.path.join(ASSETS_DIR, filename)
        if os.path.exists(filepath):
            try:
                img = Image.open(filepath).resize(size)
                return ctk.CTkImage(light_image=img, dark_image=img, size=size)
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                return text_fallback
        else:
            # Return text fallback if image doesn't exist
            return text_fallback
    
    def setup_ui(self):
        """Set up the user interface."""
        # Configure grid
        self.grid_columnconfigure(0, weight=1)  # Main content column
        self.grid_rowconfigure(0, weight=0)  # Header row - fixed height
        self.grid_rowconfigure(1, weight=1)  # Chat area - expanding
        self.grid_rowconfigure(2, weight=0)  # Input area - fixed height
        self.grid_rowconfigure(3, weight=0)  # Status bar - fixed height
        
        # ===== Header Frame =====
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        self.header_frame.grid_columnconfigure(0, weight=1)  # Title
        self.header_frame.grid_columnconfigure(1, weight=0)  # Settings
        
        # App title
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Voice Chat Assistant",
            font=("Segoe UI", 20, "bold")
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        # Theme toggle and settings in header
        self.header_buttons_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.header_buttons_frame.grid(row=0, column=1, sticky="e", padx=10)
        
        # Theme toggle button
        self.theme_button = ctk.CTkButton(
            self.header_buttons_frame,
            text="",
            image=self.icons["theme"],
            width=40,
            height=40,
            command=self.toggle_theme
        )
        self.theme_button.pack(side="left", padx=5)
        
        # Settings button
        self.settings_button = ctk.CTkButton(
            self.header_buttons_frame,
            text="",
            image=self.icons["settings"],
            width=40,
            height=40,
            command=self.open_settings
        )
        self.settings_button.pack(side="left", padx=5)
        
        # ===== Chat Display =====
        self.chat_frame = ctk.CTkFrame(self)
        self.chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_frame.grid_columnconfigure(0, weight=1)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        
        # Scrollable chat area with custom bubble styling
        self.chat_display = ScrollableChatFrame(
            self.chat_frame,
            width=300,
            corner_radius=10,
            fg_color=("gray95", "gray10"),  # Light mode, dark mode
            scrollbar_button_color=("gray70", "gray30"),
            scrollbar_button_hover_color=("gray80", "gray40")
        )
        self.chat_display.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # ===== Input Area =====
        self.input_frame = ctk.CTkFrame(self, height=120, fg_color="transparent")
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=0)
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(0, weight=0)  # Voice selection
        self.input_frame.grid_rowconfigure(1, weight=1)  # Text input and buttons
        
        # Voice selection area
        self.voice_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.voice_frame.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 5))
        
        # Voice selection label
        self.voice_label = ctk.CTkLabel(
            self.voice_frame,
            text="Assistant Voice:",
            font=("Segoe UI", 12)
        )
        self.voice_label.pack(side="left", padx=(10, 5))
        
        # Voice selection dropdown
        self.voice_var = tk.StringVar()
        self.voice_dropdown = ctk.CTkOptionMenu(
            self.voice_frame,
            variable=self.voice_var,
            values=["Loading voices..."],
            width=200,
            command=self.on_voice_selected
        )
        self.voice_dropdown.pack(side="left", padx=5)
        
        # Text input area
        self.message_frame = ctk.CTkFrame(self.input_frame, fg_color="transparent")
        self.message_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 10))
        self.message_frame.grid_columnconfigure(0, weight=1)  # Text entry
        self.message_frame.grid_columnconfigure(1, weight=0)  # Send button
        self.message_frame.grid_columnconfigure(2, weight=0)  # Mic button
        self.message_frame.grid_columnconfigure(3, weight=0)  # Reset button
        
        # Text entry field
        self.text_input = ctk.CTkEntry(
            self.message_frame,
            height=40,
            placeholder_text="Type your message here...",
            font=("Segoe UI", 12)
        )
        self.text_input.grid(row=0, column=0, sticky="ew", padx=(10, 5))
        
        # Bind Enter key to send message
        self.text_input.bind("<Return>", lambda event: self.on_send_message())
        
        # Send button
        self.send_button = ctk.CTkButton(
            self.message_frame,
            text="",
            image=self.icons["send"],
            width=40,
            height=40,
            command=self.on_send_message
        )
        self.send_button.grid(row=0, column=1, padx=5)
        
        # Microphone button
        self.mic_button = ctk.CTkButton(
            self.message_frame,
            text="",
            image=self.icons["mic"],
            width=40,
            height=40,
            fg_color=("#3B8ED0", "#1F6AA5"),  # Standard button color
            command=self.toggle_listening
        )
        self.mic_button.grid(row=0, column=2, padx=20)
        
        # Reset button
        self.reset_button = ctk.CTkButton(
            self.message_frame,
            text="",
            image=self.icons["reset"],
            width=40,
            height=40,
            fg_color="transparent",
            border_width=1,
            command=self.reset_chat
        )
        self.reset_button.grid(row=0, column=3, padx=(5, 10))
        
        # ===== Status Bar =====
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_frame.grid(row=3, column=0, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)
        
        # Progress bar for processing indication
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, mode="indeterminate")
        self.progress_bar.grid(row=0, column=1, sticky="e", padx=10, pady=5)
        self.progress_bar.grid_remove()  # Hide initially

    def toggle_listening(self):
        """Toggle voice listening on/off."""
        # Don't allow toggling while speaking
        if self.is_speaking:
            self.status_var.set("Cannot listen while speaking")
            return
            
        if self.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
    
    def start_listening(self):
        """Start voice listening."""
        try:
            # Only start if not already listening and not speaking
            if self.is_listening or self.is_speaking:
                return
                
            # Update UI
            self.mic_button.configure(
                fg_color="#E74C3C",  # Red color when listening
                image=self.icons["mic_active"]
            )
            self.is_listening = True
            
            # Calibrate noise level
            self.status_var.set("Calibrating noise level...")
            self.voice_detector.calibrate_noise()
            
            # Start voice detection
            self.voice_detector.start_monitoring(
                speech_detected_callback=self.on_speech_detected,
                speech_ended_callback=self.on_speech_ended
            )
            
            # Start pulsing animation
            self.start_mic_pulsing()
            
            # Update status
            self.status_var.set("Listening...")
            
        except Exception as e:
            messagebox.showerror("Listening Error", f"Failed to start listening: {e}")
            self.stop_listening()

    def stop_listening(self):
        """Stop voice listening."""
        try:
            # Only stop if currently listening
            if not self.is_listening:
                return
                
            # Stop mic pulsing animation
            self.stop_mic_pulsing()
            
            # Stop voice detector
            self.voice_detector.stop_monitoring()
            
            # Update UI
            self.mic_button.configure(
                fg_color=("#3B8ED0", "#1F6AA5"),  # Return to default color
                image=self.icons["mic"]
            )
            self.is_listening = False
            
            # Update status if not speaking
            if not self.is_speaking:
                self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Listening Error", f"Failed to stop listening: {e}")
    
    def on_speech_detected(self, timestamp):
        """Handle speech detection."""
        if not self.is_speaking and not self.processing_speech:
            self.after(0, lambda: self.status_var.set("Listening to speech..."))
    
    def on_speech_ended(self, duration, audio_frames):
        """Handle speech ended event."""
        if duration < 0.5 or len(audio_frames) < 10:
            self.after(0, lambda: self.status_var.set("Speech too short, ignoring."))
            self.after(100, lambda: self.status_var.set("Listening..."))
            return
            
        # Mark as processing
        self.processing_speech = True
        self.start_processing_animation()
        
        # Process in a separate thread to avoid blocking UI
        threading.Thread(target=self.process_speech, 
                         args=(audio_frames,),
                         daemon=True).start()
    
    def process_speech(self, audio_frames):
        """Process speech audio."""
        try:
            # Update status
            self.after(0, lambda: self.status_var.set("Transcribing speech..."))
            
            # Convert from float32 to int16 format for Whisper
            float32_frames = audio_frames
            int16_frames = []
            
            for frame in float32_frames:
                # Convert float32 to int16
                np_audio = np.frombuffer(frame, dtype=np.float32)
                int16_audio = (np_audio * 32767).astype(np.int16)
                int16_frames.append(int16_audio.tobytes())
            
            # Transcribe audio
            transcript, language = self.speech_to_text.transcribe_audio_frames(
                int16_frames,
                sample_rate=config.SAMPLE_RATE
            )
            
            if not transcript or transcript == "Transcription failed" or transcript == "Audio too short":
                self.after(0, lambda: self.status_var.set("No clear speech detected"))
                self.processing_speech = False
                self.after(0, self.stop_processing_animation)
                return
                
            # Update chat with transcription
            timestamp = time.strftime("%H:%M")
            self.after(0, lambda: self.chat_display.add_message("You", transcript, timestamp))
            
            # Update status
            self.after(0, lambda: self.status_var.set("Getting AI response..."))
            
            # Get response from chatbot
            response = self.chatbot.send_message(transcript)
            
            # Add to chat
            timestamp = time.strftime("%H:%M")
            self.after(0, lambda: self.chat_display.add_message("Assistant", response, timestamp))
            
            # Speak response - Important change here
            self.after(0, lambda: self.status_var.set("Speaking response..."))
            
            # Stop listening while speaking
            was_listening = self.is_listening
            if was_listening:
                self.stop_listening()
                
            # Set speaking state
            self.is_speaking = True
            
            # Speak with callback to resume listening when done
            self.tts.speak_with_callback(
                response, 
                callback=lambda: self.on_speaking_complete(was_listening)
            )
                
        except Exception as e:
            self.after(0, lambda: self.status_var.set("Error processing speech"))
            messagebox.showerror("Speech Processing Error", f"Error: {e}")
            
        finally:
            # Reset processing state
            self.processing_speech = False
            self.after(0, self.stop_processing_animation)

    def on_speaking_complete(self, was_listening):
        """Handle completion of speech output."""
        # Reset speaking state
        self.is_speaking = False
        
        # Resume listening if it was active before
        if was_listening:
            self.start_listening()
        else:
            self.after(0, lambda: self.status_var.set("Ready"))


    def initialize_components(self):
        """Initialize the voice assistant components."""
        try:
            # Initialize voice detector
            self.voice_detector = VoiceDetector(
                energy_threshold=config.ENERGY_THRESHOLD,
                silence_threshold=config.SILENCE_THRESHOLD
            )
            
            # Initialize speech-to-text with GPU acceleration and direct model path
            print("Loading Whisper model with GPU acceleration...")
            self.status_var.set("Loading speech recognition model...")
            self.update_idletasks()
            
            self.speech_to_text = SpeechToText(
                model_name=config.WHISPER_MODEL_NAME,
                model_path=config.WHISPER_MODEL_PATH
            )
            
            # Initialize text-to-speech
            self.status_var.set("Initializing text-to-speech...")
            self.update_idletasks()
            self.tts = TextToSpeech(api_key=config.ELEVEN_LABS_API_KEY)
            
            # Initialize chatbot
            self.status_var.set("Initializing AI model...")
            self.update_idletasks()
            self.chatbot = AIChatbot(
                api_key=config.GEMINI_API_KEY,
                model_name=config.GEMINI_MODEL,
                system_instruction=config.SYSTEM_INSTRUCTION
            )
            
            # Populate voice dropdown
            self.populate_voice_dropdown()
            
            # Add initial message
            self.chat_display.add_message(
                "System", 
                "Voice Chat Assistant ready. You can type a message or click the microphone button to speak."
            )
            
            self.status_var.set("Ready")
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize components: {e}")
            self.destroy()
    
    def populate_voice_dropdown(self):
        """Populate the voice selection dropdown."""
        try:
            voices = self.tts.get_voice_names()
            
            # Format voice names for dropdown
            voice_options = [f"{name}" for name, _ in voices]
            voice_ids = [voice_id for _, voice_id in voices]
            
            # Set dropdown values
            self.voice_dropdown.configure(values=voice_options)
            
            # Select first voice
            if voice_options:
                self.voice_var.set(voice_options[0])
                
                # Store voice ID mapping
                self.voice_id_map = dict(zip(voice_options, voice_ids))
                
                # Set initial voice
                self.on_voice_selected(voice_options[0])
            else:
                messagebox.showwarning("Voice Selection", "No voices available from ElevenLabs")
                
        except Exception as e:
            messagebox.showerror("Voice Loading Error", f"Failed to load voices: {e}")
    
    def on_voice_selected(self, voice_name=None):
        """Handle voice selection changed."""
        if voice_name is None:
            voice_name = self.voice_var.get()
            
        if voice_name in self.voice_id_map:
            voice_id = self.voice_id_map[voice_name]
            if self.tts.set_voice(voice_id):
                self.status_var.set(f"Voice set to: {voice_name}")
    
    def on_send_message(self):
        """Handle sending a text message."""
        message = self.text_input.get().strip()
        if not message:
            return
            
        # Clear input field
        self.text_input.delete(0, tk.END)
        
        # Add user message to chat
        timestamp = time.strftime("%H:%M")
        self.chat_display.add_message("You", message, timestamp)
        
        # Process message in a separate thread
        self.start_processing_animation()
        threading.Thread(target=self.process_message, args=(message,), daemon=True).start()
    
    def process_message(self, message):
        """Process a message using the AI chatbot and speak the response."""
        try:
            # Update status
            self.status_var.set("Processing message...")
            
            # Get response from chatbot
            response = self.chatbot.send_message(message)
            
            # Add to chat
            timestamp = time.strftime("%H:%M")
            
            # Use after to update UI from the main thread
            self.after(0, lambda: self.chat_display.add_message("Assistant", response, timestamp))
            
            # Speak response
            self.after(0, lambda: self.status_var.set("Speaking response..."))
            
            # Stop listening while speaking
            was_listening = self.is_listening
            if was_listening:
                self.stop_listening()
                
            # Set speaking state
            self.is_speaking = True
            
            # Speak with callback to resume listening when done
            self.tts.speak_with_callback(
                response, 
                callback=lambda: self.on_speaking_complete(was_listening)
            )
            
        except Exception as e:
            self.after(0, lambda: self.status_var.set("Error processing message"))
            messagebox.showerror("Processing Error", f"Error: {e}")
        finally:
            # Stop processing animation
            self.after(0, self.stop_processing_animation)

    def reset_chat(self):
        """Reset the chat history."""
        confirm_dialog = ctk.CTkInputDialog(
            text="Type 'reset' to confirm clearing chat history:",
            title="Confirm Reset"
        )
        result = confirm_dialog.get_input()
        
        if result and result.lower() == "reset":
            # Reset chatbot
            self.chatbot.reset_chat()
            
            # Clear chat display
            self.chat_display.clear()
            
            # Add system message
            self.chat_display.add_message(
                "System", 
                "Chat history has been reset. Start a new conversation."
            )
            
            # Update status
            self.status_var.set("Chat reset")
    
    def toggle_theme(self):
        """Toggle between light and dark mode."""
        if ctk.get_appearance_mode() == "Dark":
            ctk.set_appearance_mode("Light")
        else:
            ctk.set_appearance_mode("Dark")
    
    def open_settings(self):
        """Open settings dialog."""
        self.settings_window = SettingsDialog(self)
        self.settings_window.focus()
    
    def start_processing_animation(self):
        """Start the processing animation."""
        self.progress_bar.grid()
        self.progress_bar.start()
    
    def stop_processing_animation(self):
        """Stop the processing animation."""
        self.progress_bar.stop()
        self.progress_bar.grid_remove()
    
    def start_mic_pulsing(self):
        pass

    
    def stop_mic_pulsing(self):
        """Stop pulsing animation for microphone button."""
        if self.animation_after_id:
            self.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        
        # Reset button size
        self.mic_button.configure(width=40, height=40)
    
    def on_closing(self):
        """Handle window closing."""
        # Stop any ongoing processes
        if self.is_listening:
            self.stop_listening()
        
        if self.is_speaking:
            self.tts.stop_speaking()
        
        # Destroy the window
        self.destroy()


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog window."""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        
        # Configure window
        self.title("Settings")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Ensure this window stays on top of parent
        self.transient(parent)
        self.grab_set()
        
        # Create UI
        self.create_widgets()
        
        # Load current settings
        self.load_settings()
        
        # Center window
        self.center_window()
    
    def create_widgets(self):
        """Create the settings widgets."""
        # Main frame with padding
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Voice Assistant Settings",
            font=("Segoe UI", 16, "bold")
        )
        self.title_label.pack(pady=(0, 20))
        
        # Tabs
        self.tab_view = ctk.CTkTabview(self.main_frame)
        self.tab_view.pack(fill="both", expand=True)
        
        # Create tabs
        self.tab_general = self.tab_view.add("General")
        self.tab_voice = self.tab_view.add("Voice")
        self.tab_whisper = self.tab_view.add("Speech Recognition")
        self.tab_about = self.tab_view.add("About")
        
        # === General Tab ===
        self.tab_general.grid_columnconfigure(0, weight=1)
        self.tab_general.grid_columnconfigure(1, weight=1)
        
        # Theme selection
        self.theme_label = ctk.CTkLabel(self.tab_general, text="UI Theme:")
        self.theme_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.theme_var = ctk.StringVar(value=ctk.get_appearance_mode())
        self.theme_option = ctk.CTkOptionMenu(
            self.tab_general,
            values=["System", "Light", "Dark"],
            variable=self.theme_var,
            command=self.on_theme_change
        )
        self.theme_option.grid(row=0, column=1, sticky="e", padx=10, pady=10)
        
        # Export conversation
        self.export_label = ctk.CTkLabel(self.tab_general, text="Export Conversation:")
        self.export_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)
        
        self.export_button = ctk.CTkButton(
            self.tab_general,
            text="Export as Text",
            command=self.export_conversation
        )
        self.export_button.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # === Voice Tab ===
        self.tab_voice.grid_columnconfigure(0, weight=1)
        self.tab_voice.grid_columnconfigure(1, weight=1)
        
        # Voice volume
        self.volume_label = ctk.CTkLabel(self.tab_voice, text="Voice Volume:")
        self.volume_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.volume_var = ctk.DoubleVar(value=1.0)
        self.volume_slider = ctk.CTkSlider(
            self.tab_voice,
            from_=0.0,
            to=1.0,
            variable=self.volume_var,
            command=self.on_volume_change
        )
        self.volume_slider.grid(row=0, column=1, sticky="e", padx=10, pady=10)
        
        # Voice speaking rate 
        self.rate_label = ctk.CTkLabel(self.tab_voice, text="Speaking Rate:")
        self.rate_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)
        
        self.rate_var = ctk.DoubleVar(value=1.0)
        self.rate_slider = ctk.CTkSlider(
            self.tab_voice,
            from_=0.5,
            to=2.0,
            variable=self.rate_var,
            command=self.on_rate_change
        )
        self.rate_slider.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # === Whisper Tab ===
        self.tab_whisper.grid_columnconfigure(0, weight=1)
        self.tab_whisper.grid_columnconfigure(1, weight=1)
        
        # Energy threshold
        self.energy_label = ctk.CTkLabel(self.tab_whisper, text="Voice Detection Sensitivity:")
        self.energy_label.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.energy_var = ctk.DoubleVar(value=config.ENERGY_THRESHOLD)
        self.energy_slider = ctk.CTkSlider(
            self.tab_whisper,
            from_=0.005,
            to=0.05,
            variable=self.energy_var,
            command=self.on_energy_change
        )
        self.energy_slider.grid(row=0, column=1, sticky="e", padx=10, pady=10)
        
        # Silence threshold
        self.silence_label = ctk.CTkLabel(self.tab_whisper, text="Silence Timeout (seconds):")
        self.silence_label.grid(row=1, column=0, sticky="w", padx=10, pady=10)
        
        self.silence_var = ctk.DoubleVar(value=config.SILENCE_THRESHOLD)
        self.silence_slider = ctk.CTkSlider(
            self.tab_whisper,
            from_=0.5,
            to=3.0,
            variable=self.silence_var,
            command=self.on_silence_change
        )
        self.silence_slider.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        
        # === About Tab ===
        self.about_text = ctk.CTkTextbox(self.tab_about, height=200, width=400)
        self.about_text.pack(padx=10, pady=10, fill="both", expand=True)
        self.about_text.insert("1.0", """Voice Chat Assistant

Version: 1.0
        
This application combines:
• OpenAI's Whisper for speech recognition
• Google's Gemini AI for conversation
• ElevenLabs for text-to-speech

Created using CustomTkinter for a modern UI experience.

© 2024 All rights reserved.
""")
        self.about_text.configure(state="disabled")
        
        # Bottom button area
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x", pady=(20, 0))
        
        # Save button
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Settings",
            command=self.save_settings
        )
        self.save_button.pack(side="right", padx=5)
        
        # Close button
        self.close_button = ctk.CTkButton(
            self.button_frame,
            text="Close",
            command=self.destroy
        )
        self.close_button.pack(side="right", padx=5)
    
    def center_window(self):
        """Center the settings window relative to the parent."""
        self.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        width = self.winfo_width()
        height = self.winfo_height()
        
        x = parent_x + (parent_width - width) // 2
        y = parent_y + (parent_height - height) // 2
        
        self.geometry(f"+{x}+{y}")
    
    def load_settings(self):
        """Load current settings into the UI."""
        # Load any saved settings from config or use defaults
        pass
    
    def save_settings(self):
        """Save settings."""
        try:
            # Update voice detector settings
            self.parent.voice_detector.energy_threshold = self.energy_var.get()
            self.parent.voice_detector.silence_threshold = self.silence_var.get()
            
            # Update config values
            config.ENERGY_THRESHOLD = self.energy_var.get()
            config.SILENCE_THRESHOLD = self.silence_var.get()
            
            # Show confirmation
            self.save_button.configure(text="✓ Saved")
            self.after(1000, lambda: self.save_button.configure(text="Save Settings"))
            
        except Exception as e:
            messagebox.showerror("Settings Error", f"Failed to save settings: {e}")
    
    def on_theme_change(self, new_theme):
        """Handle theme change."""
        ctk.set_appearance_mode(new_theme)
    
    def on_volume_change(self, value):
        """Handle volume change."""
        # This would need implementation in the TTS module
        pass
    
    def on_rate_change(self, value):
        """Handle speech rate change."""
        # This would need implementation in the TTS module
        pass
    
    def on_energy_change(self, value):
        """Handle energy threshold change."""
        # Just update UI, actual save happens on save_settings
        pass
    
    def on_silence_change(self, value):
        """Handle silence threshold change."""
        # Just update UI, actual save happens on save_settings
        pass
    
    def export_conversation(self):
        """Export conversation history to a text file."""
        try:
            # Create file dialog
            filepath = tk.filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                title="Export Conversation"
            )
            
            if not filepath:
                return
            
            # Extract conversation from chat display
            conversation = []
            for container, sender_label, message_label, _ in self.parent.chat_display.message_widgets:
                sender = sender_label.cget("text")
                message = message_label.cget("text")
                conversation.append(f"{sender}: {message}\n")
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write("Voice Assistant Conversation\n")
                file.write("===========================\n\n")
                file.writelines(conversation)
            
            messagebox.showinfo("Export Complete", f"Conversation exported to {filepath}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export conversation: {e}")


def create_assets_directory():
    """Create the assets directory and generate minimal required assets."""
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # Check for mic icon - create a simple one if it doesn't exist
    mic_icon_path = os.path.join(ASSETS_DIR, "mic.png")
    if not os.path.exists(mic_icon_path):
        try:
            # Try to create a simple microphone icon
            from PIL import Image, ImageDraw
            
            # Create a blank image
            img = Image.new('RGBA', (24, 24), color=(0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Draw a simple microphone shape
            draw.rectangle((9, 5, 15, 15), fill=(66, 133, 244))
            draw.rectangle((7, 14, 17, 16), fill=(66, 133, 244))
            draw.rounded_rectangle((10, 16, 14, 20), radius=1, fill=(66, 133, 244))
            
            # Save the image
            img.save(mic_icon_path)
            
            # Create active version (red)
            mic_active_path = os.path.join(ASSETS_DIR, "mic_active.png")
            img_active = Image.new('RGBA', (24, 24), color=(0, 0, 0, 0))
            draw_active = ImageDraw.Draw(img_active)
            
            # Draw a simple microphone shape (red)
            draw_active.rectangle((9, 5, 15, 15), fill=(231, 76, 60))
            draw_active.rectangle((7, 14, 17, 16), fill=(231, 76, 60))
            draw_active.rounded_rectangle((10, 16, 14, 20), radius=1, fill=(231, 76, 60))
            
            # Save the active image
            img_active.save(mic_active_path)
            
        except Exception as e:
            print(f"Could not create mic icon: {e}")
            # We'll fall back to text-based buttons if this fails

def main():
    # Create assets directory
    create_assets_directory()
    
    # Create and run app
    app = ModernVoiceAssistantApp()
    app.mainloop()

if __name__ == "__main__":
    main()
    
    
    def on_speech_detected(self, timestamp):
        """Handle speech detection."""
        if not self.is_speaking and not self.processing_speech:
            self.after(0, lambda: self.status_var.set("Listening to speech..."))
    
    def on_speech_ended(self, duration, audio_frames):
        """Handle speech ended event."""
        if duration < 0.5 or len(audio_frames) < 10:
            self.after(0, lambda: self.status_var.set("Speech too short, ignoring."))
            self.after(100, lambda: self.status_var.set("Listening..."))
            return
            
        # Mark as processing
        self.processing_speech = True
        self.start_processing_animation()
        
        # Process in a separate thread to avoid blocking UI
        threading.Thread(target=self.process_speech, 
                         args=(audio_frames,),
                         daemon=True).start()
    
    def process_speech(self, audio_frames):
        """Process speech audio."""
        try:
            # Update status
            self.after(0, lambda: self.status_var.set("Transcribing speech..."))
            
            # Convert from float32 to int16 format for Whisper
            float32_frames = audio_frames
            int16_frames = []
            
            for frame in float32_frames:
                # Convert float32 to int16
                np_audio = np.frombuffer(frame, dtype=np.float32)
                int16_audio = (np_audio * 32767).astype(np.int16)
                int16_frames.append(int16_audio.tobytes())
            
            # Transcribe audio
            transcript, language = self.speech_to_text.transcribe_audio_frames(
                int16_frames,
                sample_rate=config.SAMPLE_RATE
            )
            
            if not transcript or transcript == "Transcription failed" or transcript == "Audio too short":
                self.after(0, lambda: self.status_var.set("No clear speech detected"))
                self.processing_speech = False
                self.after(0, self.stop_processing_animation)
                return
                
            # Update chat with transcription
            timestamp = time.strftime("%H:%M")
            self.after(0, lambda: self.chat_display.add_message("You", transcript, timestamp))
            
            # Update status
            self.after(0, lambda: self.status_var.set("Getting AI response..."))
            
            # Get response from chatbot
            response = self.chatbot.send_message(transcript)
            
            # Add to chat
            timestamp = time.strftime("%H:%M")
            self.after(0, lambda: self.chat_display.add_message("Assistant", response, timestamp))
            
            # Speak response
            self.after(0, lambda: self.status_var.set("Speaking response..."))
            self.is_speaking = True
            self.tts.speak(response)
            self.is_speaking = False
            
            # Update UI
            if self.is_listening:
                self.after(0, lambda: self.status_var.set("Listening..."))
            else:
                self.after(0, lambda: self.status_var.set("Ready"))
                
        except Exception as e:
            self.after(0, lambda: self.status_var.set("Error processing speech"))
            messagebox.showerror("Speech Processing Error", f"Error: {e}")
            
        finally:
            # Reset processing state
            self.processing_speech = False
            self.after(0, self.stop_processing_animation)