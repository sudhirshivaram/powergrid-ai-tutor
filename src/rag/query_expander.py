"""
Query expansion module to improve retrieval by expanding user queries.
"""

from typing import List
from llama_index.core import Settings
import sys
from pathlib import Path

# Add project root to path for config import
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config import QUERY_EXPANSION
except ImportError:
    QUERY_EXPANSION = {"max_expansions": 5}


class QueryExpander:
    """
    Expands user queries with related terms, synonyms, and acronyms
    to improve retrieval accuracy, especially for BM25 keyword search.
    """

    def __init__(self, max_expansions: int = None):
        """
        Initialize query expander.

        Args:
            max_expansions: Maximum number of expansion terms to add (uses config.py if not provided)
        """
        self.max_expansions = max_expansions if max_expansions is not None else QUERY_EXPANSION["max_expansions"]
        self.expansion_prompt_template = """You are a power systems and renewable energy expert.
Your task is to expand the user's query with related technical terms, synonyms, and acronyms.

User Query: {query}

Generate up to {max_expansions} related terms that would help find relevant information about this query.
Focus on:
1. Technical synonyms (e.g., "PV" for "solar panels")
2. Related acronyms (e.g., "BESS" for "battery energy storage")
3. Domain-specific terminology (e.g., "MPPT" for solar optimization)
4. Alternative phrasings of the same concept

Return ONLY the expansion terms, one per line, without explanations.
If the query is already very specific with technical terms, return fewer or no expansions.

Expansion terms:"""

    def expand(self, query: str) -> str:
        """
        Expand a query with related technical terms.

        Args:
            query: Original user query

        Returns:
            Expanded query string combining original + expansion terms
        """
        # Generate expansion terms using LLM
        prompt = self.expansion_prompt_template.format(
            query=query,
            max_expansions=self.max_expansions
        )

        response = Settings.llm.complete(prompt).text.strip()

        # Parse expansion terms (one per line)
        expansion_terms = [
            term.strip()
            for term in response.split('\n')
            if term.strip() and not term.strip().startswith('#')
        ]

        # Limit to max_expansions
        expansion_terms = expansion_terms[:self.max_expansions]

        # Combine original query with expansion terms
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
        else:
            expanded_query = query

        return expanded_query

    def expand_with_details(self, query: str) -> dict:
        """
        Expand query and return detailed information about the expansion.

        Args:
            query: Original user query

        Returns:
            Dictionary with original query, expansion terms, and expanded query
        """
        # Generate expansion terms using LLM
        prompt = self.expansion_prompt_template.format(
            query=query,
            max_expansions=self.max_expansions
        )

        response = Settings.llm.complete(prompt).text.strip()

        # Parse expansion terms
        expansion_terms = [
            term.strip()
            for term in response.split('\n')
            if term.strip() and not term.strip().startswith('#')
        ]

        # Limit to max_expansions
        expansion_terms = expansion_terms[:self.max_expansions]

        # Combine original query with expansion terms
        if expansion_terms:
            expanded_query = f"{query} {' '.join(expansion_terms)}"
        else:
            expanded_query = query

        return {
            'original_query': query,
            'expansion_terms': expansion_terms,
            'expanded_query': expanded_query
        }


"""
Explanation:

What Query Expansion Does:
--------------------------
1. Takes user's natural language query
2. Uses LLM to identify related technical terms
3. Adds these terms to improve keyword matching

Example:
--------
Original: "How do solar panels work?"
Expansion terms: ["photovoltaic", "PV", "solar cells", "MPPT", "inverter"]
Expanded: "How do solar panels work? photovoltaic PV solar cells MPPT inverter"

Why This Helps:
---------------
1. BM25 Keyword Search: Expanded terms increase chance of matching technical documents
2. User-Friendly: Users don't need to know exact technical terminology
3. Acronyms: Helps match acronyms like "MPPT", "BESS", "PV"
4. Synonyms: Catches different ways of expressing same concept

When It's Most Useful:
-----------------------
- User asks in natural language without technical terms
- Domain has many acronyms (power systems, renewable energy)
- Documents use specialized terminology
- Query is too general or vague

Integration with Hybrid Search:
-------------------------------
Original Query → Query Expansion → Hybrid Search (BM25 + Semantic)
- BM25 benefits most from expansion terms
- Semantic search still uses original query meaning
- Best of both: keyword matching + semantic understanding
"""
