"""
Test the complete RAG pipeline end-to-end.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_complete_pipeline():
    """Test the complete RAG pipeline with a real query."""
    
    print("Testing Complete RAG Pipeline")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Load existing vector store (we already built it in test_phase1.py)
    pipeline.load_existing()
    
    print("\n" + "=" * 70)
    print("TESTING RETRIEVAL AND GENERATION")
    print("=" * 70)
    
    # Test queries about solar energy
    queries = [
        "What is solar energy?",
        "How does solar power work?",
        "What are the benefits of solar energy?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\nQuery {i}: {query}")
        print("-" * 70)
        
        # Get answer with sources
        result = pipeline.query(query, return_sources=True)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\n\nSources used ({len(result['sources'])} chunks):")
        for source in result['sources']:
            print(f"  - Chunk {source['chunk_id']}")
            if source['score']:
                print(f"    Score: {source['score']:.4f}")
            print(f"    Text: {source['text']}")
            print()
    
    print("\n" + "=" * 70)
    print("RAG PIPELINE TEST COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    test_complete_pipeline()
