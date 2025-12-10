import cohere
from config import Config
import logging

logger = logging.getLogger(__name__)

class CohereManager:
    def __init__(self):
        # Initialize Cohere client
        self.client = cohere.Client(Config.COHERE_API_KEY)

    def embed_texts(self, texts: list[str], input_type: str = "search_document") -> list[list[float]]:
        """
        Generate embeddings for a list of texts
        """
        response = self.client.embed(
            texts=texts,
            model=Config.EMBEDDING_MODEL,
            input_type=input_type
        )
        return response.embeddings

    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using Cohere's chat model
        """
        response = self.client.chat(
            model=Config.COHERE_GENERATION_MODEL,
            message=prompt,
            max_tokens=Config.MAX_TOKENS,
            temperature=Config.TEMPERATURE
        )

        if response.text:
            return response.text.strip()
        else:
            raise Exception("No response returned from Cohere")

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a query
        """
        return self.embed_texts([query], input_type="search_query")[0]