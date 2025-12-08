"""
Test script for Phase 1 - Basic RAG components
"""

import sys
import faiss
from pathlib import Path


from llama_index.core import StorageContext, VectorStoreIndex

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import DocumentLoader
from src.data.chunkers import TextChunker

from src.data.embedders import EmbeddingManager
from src.vector_store.faiss_store import FAISSVectorStore



# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loaders import DocumentLoader


def test_document_loader():
    """Test loading the solar electric PDF"""
    
    print("Testing Document Loader...")
    print("-" * 50)
    
    # Initialize loader
    loader = DocumentLoader()
    
    # Path to your PDF
    pdf_path = "data/raw/standards/EPA02461_2010.pdf"
    
    # Load the PDF
    documents = loader.load_pdf(pdf_path)
    
    # Display results
    print(f"Successfully loaded PDF")
    print(f"Number of pages: {len(documents)}")
    print(f"\nFirst page preview (first 500 characters):")
    print("-" * 50)
    print(documents[0].text[:500])
    print("-" * 50)
    
    return documents

def test_chunker(documents):
    """Test chunking the documents"""
    
    print("\n\nTesting Text Chunker...")
    print("-" * 50)
    
    # Initialize chunker
    # chunk_size=512 tokens, chunk_overlap=50 tokens
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    
    # Chunk the documents
    nodes = chunker.chunk_documents(documents)
    
    # Get statistics
    stats = chunker.get_chunk_info(nodes)
    
    # Display results
    print(f"Chunking complete!")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Average chunk length: {stats['average_chunk_length']:.0f} characters")
    print(f"Min chunk length: {stats['min_chunk_length']}")
    print(f"Max chunk length: {stats['max_chunk_length']}")
    
    print(f"\nFirst chunk preview:")
    print("-" * 50)
    print(nodes[0].text[:300])
    print("-" * 50)
    
    return nodes


def test_embeddings_and_vector_store(nodes):
    print("\n\nTesting Embeddings and Vector Store...")
    print("-" * 50)

    # Step 1: Initialize embedding manager
    print("Step 1: Initializing embedding manager...")
    embed_manager = EmbeddingManager()
    print()

    # Step 2: Initialize your FAISSVectorStore wrapper
    print("Step 2: Initializing FAISS vector store...")
    store = FAISSVectorStore()  # Auto-detect dimension
    print("FAISS vector store initialized")
    print()

    # Step 3: Build index
    print("Step 3: Building vector index...")
    index = store.build_index(nodes)
    print()

    # Step 4: Save the index
    print("Step 4: Saving index to disk...")
    store.save()
    print()

    print("Vector store creation complete!")
    print("-" * 50)

    return store, index

if __name__ == "__main__":
    print("Running Phase 1 test script...")
    docs = test_document_loader()
    chunks = test_chunker(docs)
    vector_store, index = test_embeddings_and_vector_store(chunks)
