"""
Generation module for creating answers using LLM.
"""

from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import NodeWithScore


class Generator:
    """
    Handles answer generation using retrieved context and LLM.
    """
    
    def __init__(self, index: VectorStoreIndex):
        """
        Initialize generator with a vector index.
        
        Args:
            index: VectorStoreIndex to use for querying
        """
        self.index = index
        
        # Create query engine with prompt to prevent hallucination
        qa_prompt_str = (
            "You are an AI assistant specialized in electrical engineering, renewable energy, power systems, and smart grids.\n\n"
            "Instructions:\n"
            "1. Read the context provided below.\n"
            "2. If the question is related to electrical engineering, renewable energy, power systems, or smart grids, answer using the context - even if the context is partial or general.\n"
            "3. For broad questions (e.g., 'batteries', 'transformers'), provide what information you can find in the context.\n"
            "4. Only if the question is completely unrelated to these domains (e.g., cooking, sports, entertainment), respond with: 'I'm here to help with questions about electrical engineering, renewable energy, and power systems. This topic is outside my area of expertise, but I'd be happy to discuss solar panels, wind turbines, batteries, smart grids, or any related electrical engineering concepts. What would you like to learn about?'\n\n"
            "Context:\n"
            "{context_str}\n\n"
            "Question: {query_str}\n\n"
            "Answer:"
        )
        
        qa_prompt_template = PromptTemplate(qa_prompt_str)
        
        self.query_engine = index.as_query_engine(
            text_qa_template=qa_prompt_template
        )
    
    def generate(self, query: str) -> str:
        """
        Generate an answer to a query using RAG.
        
        Process:
        1. Retrieve relevant chunks (handled by query_engine)
        2. Format system prompt with context
        3. Call LLM to generate answer
        
        Args:
            query: User's question
            
        Returns:
            Generated answer as string
        """
        # Query engine automatically:
        # 1. Retrieves relevant chunks
        # 2. Formats prompt with context
        # 3. Calls LLM
        # 4. Returns response
        response = self.query_engine.query(query)
        
        return str(response)
    
    def generate_with_sources(self, query: str) -> dict:
        """
        Generate answer and return source information.
        
        Args:
            query: User's question
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        response = self.query_engine.query(query)
        
        # Extract source nodes (retrieved chunks)
        sources = []
        if hasattr(response, 'source_nodes'):
            for i, node in enumerate(response.source_nodes):
                source_info = {
                    'chunk_id': i + 1,
                    'score': node.score if hasattr(node, 'score') else None,
                    'text': node.text[:200] + "...",  # Preview only
                }
                # Add metadata if available
                if hasattr(node, 'metadata'):
                    source_info['metadata'] = node.metadata
                    source_info['source'] = node.metadata.get('source', 'Unknown')
                sources.append(source_info)
        
        return {
            'answer': str(response),
            'sources': sources
        }
        
        """
        Explanation:
        
        What this does:
        
        1. query_engine: LlamaIndex's built-in RAG engine
        * Automatically retrieves chunks
        * Formats prompt with context
        * Calls LLM
        2. generate(): Simple interface to get answers
        3. generate_with_sources(): Returns answer + source chunks used
        
         The generation process:
         
        * Takes query: "What is solar energy?"
        * Retrieves top 5 relevant chunks
        * Creates prompt: "Given this context: [chunks], answer: [query]"
        * LLM generates answer based on context
        
        """