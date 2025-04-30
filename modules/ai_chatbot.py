"""
AI Chatbot module using Google's Generative AI (Gemini).
Based on chatbot.py
"""

import google.generativeai as genai

class AIChatbot:
    def __init__(self, api_key, model_name="gemini-2.0-flash", system_instruction=None):
        """
        Initialize AI chatbot.
        
        Args:
            api_key: Google Generative AI API key
            model_name: Model name to use
            system_instruction: System instruction for the AI
        """
        # Configure API
        genai.configure(api_key=api_key)
        
        # Store parameters
        self.model_name = model_name
        self.system_instruction = system_instruction
        
        # Initialize model
        self.model = self._initialize_model()
        
        # Initialize chat with memory
        self.chat = self.model.start_chat(history=[])
        
    def _initialize_model(self):
        """Initialize and return the Generative AI model."""
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.system_instruction
            )
            return model
        except Exception as e:
            print(f"Error initializing Generative AI model: {e}")
            raise
    
    def send_message(self, user_input):
        """
        Send a message to the chatbot and get the response.
        
        Args:
            user_input: User's message
            
        Returns:
            Bot's response text
        """
        if not user_input.strip():
            return "I didn't catch that. Could you please say something?"
            
        try:
            # Send user message
            response = self.chat.send_message(user_input)
            
            # Get bot response
            return response.text
        except Exception as e:
            print(f"Error communicating with AI: {e}")
            return f"Sorry, I encountered a problem processing your message. Please try again."
    
    def reset_chat(self):
        """Reset the chat history."""
        try:
            self.chat = self.model.start_chat(history=[])
            return True
        except Exception as e:
            print(f"Error resetting chat: {e}")
            return False

# Example usage
if __name__ == "__main__":
    from config import GEMINI_API_KEY, GEMINI_MODEL, SYSTEM_INSTRUCTION
    
    chatbot = AIChatbot(
        api_key=GEMINI_API_KEY,
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_INSTRUCTION
    )
    
    # Test message
    response = chatbot.send_message("Hello, who are you?")
    print(f"Bot: {response}")