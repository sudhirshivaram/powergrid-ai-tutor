"""
ArXiv paper collector for electrical engineering and renewable energy topics.
"""

import os
import time
from pathlib import Path
from typing import List, Dict
import arxiv
import json


class ArxivCollector:
    """
    Collects research papers from ArXiv on electrical engineering topics.
    """
    
    def __init__(self, output_dir: str = "data/raw/papers"):
        """
        Initialize ArXiv collector.
        
        Args:
            output_dir: Directory to save downloaded papers
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file to track downloaded papers
        self.metadata_file = self.output_dir / "arxiv_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"papers": []}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def search_and_download(self, 
                           query: str, 
                           max_results: int = 20,
                           category: str = None) -> List[Dict]:
        """
        Search ArXiv and download papers.
        
        Args:
            query: Search query
            max_results: Maximum number of papers to download
            category: ArXiv category filter (e.g., "eess.SY")
            
        Returns:
            List of downloaded paper metadata
        """
        print(f"Searching ArXiv for: {query}")
        print(f"Max results: {max_results}")
        print("-" * 60)
        
        # Build search query
        search_query = query
        if category:
            search_query = f"cat:{category} AND {query}"
        
        # Search ArXiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        downloaded_papers = []
        
        for i, result in enumerate(client.results(search), 1):
            print(f"\n[{i}/{max_results}] Processing: {result.title}")
            
            # Check if already downloaded
            paper_id = result.entry_id.split('/')[-1]
            if any(p['id'] == paper_id for p in self.metadata['papers']):
                print(f"  Already downloaded, skipping...")
                continue
            
            # Download PDF
            try:
                pdf_filename = f"{paper_id.replace('/', '_')}.pdf"
                pdf_path = self.output_dir / pdf_filename
                
                print(f"  Downloading to: {pdf_filename}")
                result.download_pdf(filename=str(pdf_path))
                
                # Save metadata
                paper_metadata = {
                    "id": paper_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": result.published.isoformat(),
                    "categories": result.categories,
                    "summary": result.summary,
                    "pdf_url": result.pdf_url,
                    "local_path": pdf_filename
                }
                
                self.metadata['papers'].append(paper_metadata)
                downloaded_papers.append(paper_metadata)
                
                print(f"  Downloaded successfully")
                
                # Rate limiting to be respectful
                time.sleep(3)
                
            except Exception as e:
                print(f"  Error downloading: {e}")
                continue
        
        # Save metadata
        self._save_metadata()
        
        print("\n" + "=" * 60)
        print(f"Downloaded {len(downloaded_papers)} new papers")
        print(f"Total papers in collection: {len(self.metadata['papers'])}")
        print("=" * 60)
        
        return downloaded_papers
    
    def collect_electrical_engineering_papers(self, max_per_topic: int = 10):
        """
        Collect papers on key electrical engineering topics.
        
        Args:
            max_per_topic: Maximum papers per topic
        """
        topics = [
            {
                "query": "solar energy photovoltaic",
                "category": "physics.app-ph",
                "max_results": max_per_topic
            },
            {
                "query": "power grid renewable energy",
                "category": "eess.SY",
                "max_results": max_per_topic
            },
            {
                "query": "wind energy power generation",
                "category": "physics.app-ph",
                "max_results": max_per_topic
            },
            {
                "query": "battery energy storage",
                "category": "physics.app-ph",
                "max_results": max_per_topic
            },
            {
                "query": "smart grid power systems",
                "category": "eess.SY",
                "max_results": max_per_topic
            }
        ]
        
        print("Starting ArXiv paper collection for electrical engineering topics")
        print("=" * 60)
        
        all_papers = []
        
        for i, topic in enumerate(topics, 1):
            print(f"\n\nTopic {i}/{len(topics)}: {topic['query']}")
            print("=" * 60)
            
            papers = self.search_and_download(
                query=topic['query'],
                max_results=topic['max_results'],
                category=topic['category']
            )
            
            all_papers.extend(papers)
            
            # Pause between topics
            if i < len(topics):
                print(f"\nPausing for 5 seconds before next topic...")
                time.sleep(5)
        
        print("\n\n" + "=" * 60)
        print("COLLECTION COMPLETE")
        print(f"Total new papers downloaded: {len(all_papers)}")
        print(f"Total papers in collection: {len(self.metadata['papers'])}")
        print("=" * 60)
        
        return all_papers


def main():
    """Main function to run the collector."""
    collector = ArxivCollector()
    
    # Collect papers on electrical engineering topics
    # Start with 10 papers per topic (50 total)
    collector.collect_electrical_engineering_papers(max_per_topic=10)


if __name__ == "__main__":
    main()
    
"""
    Explanation:
    
    This script:
    
    1.Searches ArXiv for electrical engineering topics
    2.Downloads PDFs automatically
    3.Saves metadata (title, authors, abstract)
    4.Tracks what's already downloaded
    5.Includes rate limiting to be respectful
     
 """