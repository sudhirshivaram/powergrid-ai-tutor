"""
Test the ArXiv collector with a small sample.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.data_collection.collect_arxiv_papers import ArxivCollector


def test_small_collection():
    """Test with just 2 papers."""
    
    print("Testing ArXiv Collector")
    print("=" * 60)
    
    collector = ArxivCollector()
    
    # Download just 2 papers on solar energy
    papers = collector.search_and_download(
        query="solar energy photovoltaic",
        max_results=2,
        category="physics.app-ph"
    )
    
    print("\n\nTest Results:")
    print("-" * 60)
    print(f"Papers downloaded: {len(papers)}")
    
    for paper in papers:
        print(f"\nTitle: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'][:3])}")
        print(f"File: {paper['local_path']}")


if __name__ == "__main__":
    test_small_collection()