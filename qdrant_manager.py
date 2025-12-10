from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self):
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
            port=Config.QDRANT_PORT
        )
        self.collection_name = Config.COLLECTION_NAME
    
    def create_collection(self):
        """Create the collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists")
        except:
            # Create collection with vector size of 4096 (Cohere's embedding dimension)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=4096, distance=models.Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection_name}'")
    
    def insert_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: List[str]):
        """Insert vectors into the collection"""
        self.client.upsert(
            collection_name=self.collection_name,
            points=models.Batch(
                ids=ids,
                vectors=vectors,
                payloads=payloads
            )
        )
        logger.info(f"Inserted {len(vectors)} vectors into collection")
    
    def search_vectors(self, query_vector: List[float], limit: int = 5):
        """Search for similar vectors in the collection"""
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return search_results
    
    def delete_collection(self):
        """Delete the collection (useful for testing/resetting)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self):
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None