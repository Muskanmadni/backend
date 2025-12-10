import logging
from typing import List, Tuple
import uuid
from config import Config
from qdrant_manager import QdrantManager
from document_processor import DocumentProcessor
from cohere_manager import CohereManager

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.document_processor = DocumentProcessor()
        self.cohere_manager = CohereManager()

        # Initialize Gemini manager if available
        try:
            from gemini_manager import GeminiManager
            self.gemini_manager = GeminiManager()
            self.gemini_available = True
        except ImportError:
            logger.warning("Gemini manager not available")
            self.gemini_available = False
            self.gemini_manager = None

    def process_and_store_document(self, file_path: str, filename: str) -> dict:
        """
        Process a document and store its embeddings in Qdrant
        """
        try:
            # Process document into chunks
            chunks = self.document_processor.process_document(file_path, filename)

            # Prepare data for Qdrant
            payloads = []
            vectors = []
            ids = []

            for i, chunk in enumerate(chunks):
                # Generate embedding using Cohere
                embedding = self.cohere_manager.embed_texts([chunk], input_type="search_document")[0]

                payloads.append({
                    "content": chunk,
                    "source": filename,
                })
                vectors.append(embedding)
                ids.append(str(uuid.uuid4()))  # Using UUIDs for proper Qdrant point IDs

            # Upload to Qdrant
            self.qdrant_manager.insert_vectors(vectors, payloads, ids)

            return {
                "message": f"Successfully processed {filename}",
                "chunks_indexed": len(chunks)
            }

        except Exception as e:
            logger.error(f"Error in process_and_store_document: {str(e)}")
            raise e

    def retrieve_and_generate(self, query: str, use_gemini: bool = False) -> Tuple[str, List[str]]:
        """
        Retrieve relevant documents and generate a response
        """
        try:
            # Generate embedding for the query using Cohere (for retrieval)
            query_embedding = self.cohere_manager.embed_query(query)

            # Search in Qdrant for relevant documents
            search_results = self.qdrant_manager.search_vectors(query_embedding, limit=5)

            # Extract relevant context from search results
            context_parts = []
            sources = []
            for result in search_results:
                if result.payload:
                    content = result.payload.get("content", "")
                    source = result.payload.get("source", "Unknown")
                    context_parts.append(content)
                    if source not in sources:
                        sources.append(source)

            # Format context for the model
            context = "\n\n".join(context_parts)

            # If no context found, return a default response
            if not context:
                return "I couldn't find any relevant information to answer your question. Please try uploading some documents first.", []

            # Prepare the prompt
            prompt = f"""
            You are a helpful AI assistant. Use the following context to answer the user's question.
            If the context doesn't contain the information needed to answer the question, say so.

            Context:
            {context}

            Question: {query}

            Answer:
            """

            # Choose the model based on the parameter
            if use_gemini and self.gemini_available:
                # Generate response using Gemini
                answer = self.gemini_manager.generate_response(prompt)
            else:
                # Generate response using Cohere (default)
                answer = self.cohere_manager.generate_response(prompt)

            return answer, sources

        except Exception as e:
            logger.error(f"Error in retrieve_and_generate: {str(e)}")
            raise e