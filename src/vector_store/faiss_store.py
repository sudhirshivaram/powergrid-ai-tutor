"""
FAISS vector store management for storing and querying embeddings.
"""

import os
from typing import List
from pathlib import Path
import faiss

from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.faiss import FaissVectorStore


class FAISSVectorStore:
 def __init__(self, dimension: int = None, persist_dir: str = "data/vector_stores/faiss"):
 """
 Initialize FAISS vector store.

 Args:
 dimension: Embedding dimension (will be auto-detected if None)
 persist_dir: Directory to save the vector store
 """
 self.dimension = dimension
 self.persist_dir = Path(persist_dir)

 # FAISS index will be created in build_index after dimension is known
 self.faiss_index = None
 self.vector_store = None

 # Placeholder for the index object built later
 self.index = None

 # Ensure persistence directory exists
 os.makedirs(self.persist_dir, exist_ok=True)

 def build_index(self, nodes: List[Document]) -> VectorStoreIndex:
 """Build vector index from document nodes."""
 from llama_index.core import Settings

 print(f"Building FAISS index from {len(nodes)} nodes...")

 # Auto-detect embedding dimension if not provided
 if self.dimension is None:
 print("Auto-detecting embedding dimension...")
 test_embedding = Settings.embed_model.get_text_embedding("test")
 self.dimension = len(test_embedding)
 print(f"Detected embedding dimension: {self.dimension}")

 # Create FAISS index with correct dimension
 self.faiss_index = faiss.IndexFlatL2(self.dimension)

 # Create LlamaIndex vector store wrapper
 self.vector_store = FaissVectorStore(self.faiss_index)

 storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

 # Build index from nodes
 self.index = VectorStoreIndex.from_documents(
 documents=nodes,
 storage_context=storage_context,
 show_progress=True,
 )

 print(f"Index built successfully with {self.faiss_index.ntotal} vectors")
 return self.index


 def save(self):
 """Save the FAISS index to disk for later use."""
 if self.index is None:
 raise ValueError("No index to save. Build index first.")

 faiss_file = self.persist_dir / "index.faiss"
 faiss.write_index(self.faiss_index, str(faiss_file))

 # Save LlamaIndex metadata
 self.index.storage_context.persist(persist_dir=str(self.persist_dir))
 print(f"Index saved to {self.persist_dir}")

 def load(self) -> VectorStoreIndex:
 """Load a previously saved FAISS index from disk."""
 from llama_index.core import load_index_from_storage

 faiss_file = self.persist_dir / "index.faiss"
 if not faiss_file.exists():
 raise FileNotFoundError(f"No saved index found at {faiss_file}")

 loaded_faiss_index = faiss.read_index(str(faiss_file))

 # Re-wrap the loaded FAISS index
 self.vector_store = FaissVectorStore(loaded_faiss_index)

 # Load the full index with all metadata and text
 storage_context = StorageContext.from_defaults(
 vector_store=self.vector_store,
 persist_dir=str(self.persist_dir),
 )

 self.index = load_index_from_storage(storage_context)
 self.faiss_index = loaded_faiss_index

 print(f"Index loaded from {self.persist_dir}")
 print(f"Total vectors: {loaded_faiss_index.ntotal}")
 return self.index

 def get_index(self) -> VectorStoreIndex:
 """Return the current index."""
 if self.index is None:
 raise ValueError("No index available. Build or load an index first.")
 return self.index
