"""
Startup script for the RAG Chatbot
"""
import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting RAG Chatbot API server...")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)