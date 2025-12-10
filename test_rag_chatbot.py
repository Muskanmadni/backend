"""
Test script for the RAG Chatbot with both Cohere and Gemini models
"""
import asyncio
import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"

def test_api_endpoints():
    print("Testing RAG Chatbot API endpoints...")

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"   ✓ Health check passed: {response.json()}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Health check error: {e}")

    # Test root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print(f"   ✓ Root endpoint working: {response.json()}")
        else:
            print(f"   ✗ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Root endpoint error: {e}")

def test_document_upload():
    print("\n3. Testing document upload...")
    try:
        # Upload the sample document
        with open("sample_document.txt", "rb") as f:
            files = {"file": ("sample_document.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload/", files=files)

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Document uploaded successfully: {result}")
            return True
        else:
            print(f"   ✗ Document upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Document upload error: {e}")
        return False

def test_cohere_chat():
    print("\n4. Testing Cohere chat functionality...")
    try:
        # Send a query that should be answerable based on the sample document
        query_data = {
            "message": "What is a RAG system and how does it work?",
            "history": [],
            "use_gemini": False  # Use Cohere
        }

        response = requests.post(
            f"{BASE_URL}/chat/",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_data)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Cohere chat response received:")
            print(f"     Message: {result['message'][:200]}...")
            print(f"     Sources: {result['documents']}")
            return True
        else:
            print(f"   ✗ Cohere chat failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Cohere chat error: {e}")
        return False

def test_gemini_chat():
    print("\n5. Testing Gemini chat functionality...")
    try:
        # Send a query that should be answerable based on the sample document
        query_data = {
            "message": "Explain how RAG systems work with external knowledge sources.",
            "history": [],
            "use_gemini": True  # Use Gemini
        }

        response = requests.post(
            f"{BASE_URL}/chat/",
            headers={"Content-Type": "application/json"},
            data=json.dumps(query_data)
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Gemini chat response received:")
            print(f"     Message: {result['message'][:200]}...")
            print(f"     Sources: {result['documents']}")
            return True
        else:
            print(f"   ✗ Gemini chat failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ✗ Gemini chat error: {e}")
        return False

def main():
    print("Starting RAG Chatbot tests...")

    # First test the basic endpoints
    test_api_endpoints()

    # Then test document upload (this requires the server to be running)
    upload_success = test_document_upload()

    if upload_success:
        # Test Cohere chat functionality
        cohere_success = test_cohere_chat()

        # Test Gemini chat functionality
        gemini_success = test_gemini_chat()

        if cohere_success and gemini_success:
            print("\n✓ All tests passed! Both Cohere and Gemini models are working correctly.")
        else:
            print(f"\n⚠ Some tests failed. Cohere: {'✓' if cohere_success else '✗'}, Gemini: {'✓' if gemini_success else '✗'}")
    else:
        print("\nSkipping chat tests since document upload failed.")

    print("\nTests completed!")

if __name__ == "__main__":
    main()