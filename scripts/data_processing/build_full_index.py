"""
Build vector store from all collected documents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.loaders import DocumentLoader
from src.data.chunkers import TextChunker
from src.data.embedders import EmbeddingManager
from src.vector_store.faiss_store import FAISSVectorStore


def build_full_index():
 """Build vector index from all PDFs."""
 
 print("Building Full Knowledge Base for PowerGrid AI Tutor")
 print("=" * 70)
 
 # Initialize components
 print("\n1. Initializing components...")
 loader = DocumentLoader()
 chunker = TextChunker(chunk_size=512, chunk_overlap=50)
 embed_manager = EmbeddingManager()
 
 # Load all PDFs
 print("\n2. Loading all PDF documents...")
 
 # Load from papers directory
 papers_dir = Path("data/raw/papers")
 standards_dir = Path("data/raw/standards")
 
 all_documents = []
 
 # Load ArXiv papers
 if papers_dir.exists():
 print(f" Loading papers from {papers_dir}...")
 papers = loader.load_directory(str(papers_dir))
 all_documents.extend(papers)
 print(f" Loaded {len(papers)} documents from papers")
 
 # Load standards/original PDFs
 if standards_dir.exists():
 print(f" Loading documents from {standards_dir}...")
 standards = loader.load_directory(str(standards_dir))
 all_documents.extend(standards)
 print(f" Loaded {len(standards)} documents from standards")
 
 print(f"\n Total documents loaded: {len(all_documents)}")
 
 # Chunk documents
 print("\n3. Chunking documents...")
 nodes = chunker.chunk_documents(all_documents)
 stats = chunker.get_chunk_info(nodes)
 
 print(f" Total chunks created: {stats['total_chunks']}")
 print(f" Average chunk length: {stats['average_chunk_length']:.0f} characters")
 
 # Build vector store
 print("\n4. Building vector store (this may take a few minutes)...")
 vector_store = FAISSVectorStore(persist_dir="data/vector_stores/faiss_full")
 index = vector_store.build_index(nodes)
 
 # Save
 print("\n5. Saving vector store...")
 vector_store.save()
 
 print("\n" + "=" * 70)
 print("KNOWLEDGE BASE BUILD COMPLETE!")
 print("=" * 70)
 print(f"Documents indexed: {len(all_documents)}")
 print(f"Total chunks: {stats['total_chunks']}")
 print(f"Vector store saved to: data/vector_stores/faiss_full")
 print("=" * 70)


if __name__ == "__main__":
 build_full_index()
 
"""
Explanation:

This script:

 1.Loads all PDFs from both papers/ and standards/ directories
 2.Chunks them all
 3.Generates embeddings (locally, no API calls)
 4.Builds a new vector store at data/vector_stores/faiss_full
 5.Keeps your original small test index intact

""" 