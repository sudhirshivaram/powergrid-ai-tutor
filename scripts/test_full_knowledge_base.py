"""
Test the full knowledge base with various queries.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


def test_full_knowledge_base():
    """Test the expanded knowledge base with diverse queries."""
    
    print("Testing Full PowerGrid AI Tutor Knowledge Base")
    print("=" * 70)
    
    # Initialize pipeline with full index
    pipeline = RAGPipeline()
    pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")
    
    # Test queries covering different topics
    queries = [
        "What are the main challenges in solar energy systems?",
        "How does wind energy integration affect power grid stability?",
        "What are the latest advances in battery energy storage?",
        "Explain smart grid technology and its benefits",
        "What is the role of renewable energy in modern power systems?"
    ]
    
    print("\n" + "=" * 70)
    print("TESTING QUERIES")
    print("=" * 70)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\nQuery {i}: {query}")
        print("-" * 70)
        
        # Get answer with sources
        result = pipeline.query(query, return_sources=True)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\n\nSources ({len(result['sources'])} chunks retrieved):")
        for source in result['sources'][:3]:  # Show top 3 sources
            print(f"  - Chunk {source['chunk_id']}")
            if source['score']:
                print(f"    Similarity Score: {source['score']:.4f}")
            print(f"    Preview: {source['text'][:150]}...")
            print()
    
    print("\n" + "=" * 70)
    print("TESTING COMPLETE!")
    print("=" * 70)
    print("\nYour PowerGrid AI Tutor now has knowledge from 50 research papers")
    print("across solar, wind, battery, and smart grid topics!")
    print("=" * 70)


if __name__ == "__main__":
    test_full_knowledge_base()