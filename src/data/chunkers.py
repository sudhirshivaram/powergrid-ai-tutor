"""

 Chunking splits long documents into smaller pieces. This is critical because:
 * LLMs have token limits
 * Smaller chunks = more precise retrieval
 * Better matching between query and context

"""

"""
Text chunking strategies for splitting documents into smaller pieces.
"""

from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import CHUNKING
except ImportError:
    # Fallback if config.py not found
    CHUNKING = {"chunk_size": 512, "chunk_overlap": 50}


class TextChunker:
    """
    Handles splitting documents into chunks for embedding and retrieval.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize the chunker with specified parameters.
        
        Args:
            chunk_size: Maximum number of tokens per chunk (uses config.py if not provided)
            chunk_overlap: Number of tokens to overlap between chunks (uses config.py if not provided)
        """
        # Use config values if not provided
        self.chunk_size = chunk_size if chunk_size is not None else CHUNKING["chunk_size"]
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNKING["chunk_overlap"]
        
        # SentenceSplitter from LlamaIndex
        # Splits on sentence boundaries for better semantic coherence
        self.splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks and enrich with metadata.

        Args:
            documents: List of Document objects to chunk

        Returns:
            List of chunked Document objects (nodes) with metadata
        """
        # get_nodes_from_documents splits and preserves metadata
        nodes = self.splitter.get_nodes_from_documents(documents)

        # Add metadata to each chunk
        for node in nodes:
            # Extract from source document
            source_file = node.metadata.get('file_name', 'unknown')

            # Infer topic from filename or content
            topic = self._infer_topic(source_file, node.text)

            # Add enriched metadata
            node.metadata['source'] = source_file
            node.metadata['topic'] = topic

        return nodes

    def _infer_topic(self, filename: str, text: str) -> str:
        """
        Infer the topic of a chunk based on filename and content keywords.

        Args:
            filename: Source file name
            text: Chunk text content

        Returns:
            Inferred topic (solar, wind, battery, grid, or general)
        """
        # Combine filename and text for keyword matching
        content = (filename + " " + text).lower()

        # Define topic keywords (ordered by specificity)
        topic_keywords = {
            'solar': ['solar', 'photovoltaic', 'pv', 'mppt', 'inverter', 'panel'],
            'wind': ['wind', 'turbine', 'rotor', 'blade', 'nacelle'],
            'battery': ['battery', 'storage', 'bess', 'energy storage', 'lithium'],
            'grid': ['grid', 'power system', 'transmission', 'distribution', 'substation']
        }

        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content)
            if score > 0:
                topic_scores[topic] = score

        # Return topic with highest score, or 'general' if no matches
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def get_chunk_info(self, nodes: List[Document]) -> dict:
        """
        Get statistics about the chunks created.
        
        Args:
            nodes: List of chunked documents
        
        Returns:
            Dictionary with chunk statistics
        """
        total_chunks = len(nodes)
        
        # Calculate average chunk length
        chunk_lengths = [len(node.text) for node in nodes]
        avg_length = sum(chunk_lengths) / total_chunks if total_chunks > 0 else 0
        
        return {
            "total_chunks": total_chunks,
            "average_chunk_length": avg_length,
            "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0
        }

"""

 What this code does:
 * SentenceSplitter: LlamaIndex's intelligent splitter that respects sentence boundaries
 * chunk_size: Target size for each chunk (512 tokens = roughly 1-2 paragraphs)
 * chunk_overlap: Overlapping text between chunks prevents losing context at boundaries
 * get_chunk_info: Utility method to see chunking statistics

 Why these parameters:
 
 * 512 tokens: Good balance between context and precision
 * 50 overlap: Ensures continuity between chunks
 * Sentence boundaries: Better semantic coherence than arbitrary splits

"""