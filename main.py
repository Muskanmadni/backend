import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tempfile
from config import Config
from rag_service import RAGService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot using FastAPI, Cohere, and Qdrant",
    version="1.0.0"
)

# Initialize RAG service
rag_service = RAGService()

# Request/Response models
class MessageRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []
    use_gemini: bool = False  # Flag to choose between Cohere and Gemini

class DocumentResponse(BaseModel):
    message: str
    documents: List[str]

@app.on_event("startup")
def startup_event():
    logger.info("Starting up RAG Chatbot API")
    # Initialize Qdrant collection on startup
    from qdrant_manager import QdrantManager
    qdrant_manager = QdrantManager()
    qdrant_manager.create_collection()

@app.get("/")
def read_root():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT) to be indexed for RAG
    """
    try:
        # Validate file is provided
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")

        # Validate file size (optional but recommended)
        # Move to the end to get file size
        await file.seek(0, 2)  # Seek to end of file
        file_size = await file.tell()  # Get current position (file size)
        await file.seek(0)  # Reset to beginning of file

        if file_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process and store document using RAG service
        result = rag_service.process_and_store_document(temp_path, file.filename)

        # Clean up temp file
        os.unlink(temp_path)

        return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/chat/", response_model=DocumentResponse)
async def chat_with_rag(request: MessageRequest):
    """
    Chat endpoint with RAG functionality
    """
    try:
        query = request.message

        # Use RAG service to retrieve and generate response
        # Pass the use_gemini flag to determine which model to use
        answer, sources = rag_service.retrieve_and_generate(query, use_gemini=request.use_gemini)

        return DocumentResponse(
            message=answer,
            documents=sources
        )

    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during chat: {str(e)}")

@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    from qdrant_manager import QdrantManager
    qdrant_manager = QdrantManager()
    qdrant_info = qdrant_manager.get_collection_info()
    if qdrant_info:
        return {"status": "healthy", "qdrant_collection": Config.COLLECTION_NAME, "points_count": qdrant_info.points_count}
    else:
        return {"status": "degraded", "qdrant_collection": Config.COLLECTION_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
