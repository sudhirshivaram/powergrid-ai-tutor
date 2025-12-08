"""
Document loaders for loading various file formats.
This module handles loading PDFs and extracting text content.
"""

from pathlib import Path
from typing import List
from llama_index.core import Document
from llama_index.readers.file import PDFReader


class DocumentLoader:
    """
    Loads documents from various file formats.
    Currently supports PDF files.
    """
    
    def __init__(self):
        # Initialize PDF reader from LlamaIndex
        self.pdf_reader = PDFReader()
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return LlamaIndex Document objects.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects containing the PDF content
        """
        # Convert string path to Path object for better path handling
        pdf_path = Path(file_path)
        
        # Check if file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Load PDF and extract text
        # PDFReader returns a list of Document objects, one per page
        documents = self.pdf_reader.load_data(file=pdf_path)
        
        return documents
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all Document objects from all PDFs
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all PDF files in directory
        pdf_files = list(dir_path.glob("*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {directory_path}")
        
        # Load all PDFs
        all_documents = []
        for pdf_file in pdf_files:
            documents = self.load_pdf(str(pdf_file))
            all_documents.extend(documents)
        
        return all_documents
    
    
    """
    What this code does:
    1. Imports: Uses LlamaIndex's built-in PDFReader which handles PDF parsing for us
    2. DocumentLoader class: Main class to load documents
    3. load_pdf method: Loads a single PDF file and returns Document objects
    4. load_directory method: Loads all PDFs from a folder (useful later when you have multiple papers)
    
Key concepts:

 * Document object: LlamaIndex's standard format - contains text + metadata
 * PDFReader: Automatically extracts text from PDF, handles different PDF formats
 * Path handling: Uses pathlib for cross-platform compatibility
    
    """