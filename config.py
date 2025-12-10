import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application configuration
class Config:
    # Cohere configuration
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY environment variable is required")

    # Gemini configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is required")

    # Qdrant configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "localhost")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    # Collection configuration
    COLLECTION_NAME = "documents"

    # Document processing
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Model configuration
    EMBEDDING_MODEL = "embed-english-v3.0"
    COHERE_GENERATION_MODEL = "command-r-plus"
    GEMINI_MODEL = "gemini-2.5-flash"
    MAX_TOKENS = 500
    TEMPERATURE = 0.3