import google.generativeai as genai
from config import Config
import logging

logger = logging.getLogger(__name__)

class GeminiManager:
    def __init__(self):
        # Initialize Gemini client
        genai.configure(api_key=Config.GEMINI_API_KEY)
        
        # Initialize the model
        self.model = genai.GenerativeModel(
            model_name=Config.GEMINI_MODEL,
            generation_config={
                "temperature": Config.TEMPERATURE,
                "max_output_tokens": Config.MAX_TOKENS,
            }
        )
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the Gemini model
        """
        try:
            response = self.model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                raise Exception("No text returned from Gemini model")
                
        except Exception as e:
            logger.error(f"Error generating response with Gemini: {str(e)}")
            raise e