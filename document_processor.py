import tempfile
import os
from typing import List
from config import Config
from pypdf import PdfReader

class DocumentProcessor:
    def __init__(self):
        pass

    def process_document(self, file_path: str, filename: str) -> List[str]:
        """
        Process a document and return list of text chunks
        """
        # Load document based on file type
        if filename.lower().endswith('.pdf'):
            content = self._extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
        else:
            raise ValueError(f"Unsupported file type: {filename}. Please upload PDF or TXT files.")

        # Split the content into chunks
        chunks = self._split_text(content)

        return chunks

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file
        """
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        """
        chunk_size = Config.CHUNK_SIZE
        chunk_overlap = Config.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this is not the last chunk and we have overlap
            if end < len(text) and chunk_overlap > 0:
                # Try to find a sentence boundary within the overlap region
                overlap_start = end - chunk_overlap
                chunk_text = text[start:end]

                # Look for a good break point (like sentence or paragraph boundary)
                break_found = False
                for i in range(min(chunk_overlap, len(chunk_text)), 0, -1):
                    if chunk_text[-i] in '.!?;':
                        end = start + len(chunk_text) - i
                        break_found = True
                        break
                    elif chunk_text[-i] == '\n':
                        end = start + len(chunk_text) - i
                        break_found = True
                        break

                if break_found:
                    chunks.append(text[start:end])
                    start = end - chunk_overlap
                else:
                    chunks.append(text[start:end])
                    start = end - chunk_overlap
            else:
                chunks.append(text[start:end])
                start = end

        # Filter out empty chunks
        return [chunk.strip() for chunk in chunks if chunk.strip()]