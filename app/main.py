"""
Gradio UI for PowerGrid AI Tutor with API Key Input Support.
Designed for HuggingFace Space deployment and certification requirements.
"""

import sys
import os
import time
from pathlib import Path
import gradio as gr
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.pipeline import RAGPipeline


class PowerGridTutorUI:
    """
    Gradio interface for the PowerGrid AI Tutor with API key support.
    """
    
    def __init__(self):
        """
        Initialize the UI without loading the pipeline yet.
        Pipeline will be created when user provides API key.
        """
        print("PowerGrid AI Tutor UI initialized")
        self.pipeline = None
        self.available_sources = []
        
    def initialize_pipeline(self, 
                           api_key: str,
                           llm_provider: str,
                           use_reranking: bool,
                           use_hybrid: bool,
                           use_query_expansion: bool) -> str:
        """
        Initialize or reinitialize the RAG pipeline with user settings.
        
        Args:
            api_key: API key for Gemini (if using Gemini)
            llm_provider: "gemini" or "ollama"
            use_reranking: Enable LLM reranking
            use_hybrid: Enable hybrid search
            use_query_expansion: Enable query expansion
            
        Returns:
            Status message
        """
        try:
            # Validate API key for Gemini
            if llm_provider == "gemini":
                if not api_key or not api_key.strip():
                    return "❌ Error: Gemini requires an API key. Please enter your Google Gemini API key above."
                # Basic validation - Gemini keys typically start with "AI"
                if not api_key.startswith("AI"):
                    return "⚠️ Warning: Gemini API keys usually start with 'AI'. Please verify your key."
            
            print(f"\n{'='*70}")
            print("Initializing PowerGrid AI Tutor...")
            print(f"{'='*70}")
            print(f"LLM Provider: {llm_provider.upper()}")
            print(f"Query Expansion: {use_query_expansion}")
            print(f"Hybrid Search: {use_hybrid}")
            print(f"Reranking: {use_reranking}")
            
            # Create pipeline with user's API key
            self.pipeline = RAGPipeline(
                use_reranking=use_reranking,
                use_hybrid=use_hybrid,
                use_query_expansion=use_query_expansion,
                llm_provider=llm_provider,
                api_key=api_key if llm_provider == "gemini" else None
            )
            
            # Load existing vector store
            self.pipeline.load_existing(persist_dir="data/vector_stores/faiss_full")
            
            # Get available sources
            self.available_sources = self._get_available_sources()
            
            features = []
            if use_query_expansion:
                features.append("Query Expansion")
            if use_hybrid:
                features.append("Hybrid Search")
            if use_reranking:
                features.append("Reranking")
            
            features_str = " + ".join(features) if features else "Basic RAG"
            
            success_msg = f"""
✅ **PowerGrid AI Tutor Ready!**

**Configuration:**
- LLM: {llm_provider.upper()}
- Features: {features_str}
- Knowledge Base: 50 papers, 2,166 chunks
- Available Sources: {len(self.available_sources)} papers

You can now ask questions using the chat interface below!
            """
            
            print(f"\n{'='*70}")
            print("Pipeline Ready!")
            print(f"{'='*70}\n")
            
            return success_msg
            
        except Exception as e:
            error_msg = f"❌ Error initializing pipeline: {str(e)}"
            print(error_msg)
            return error_msg

    def _get_available_sources(self):
        """Get unique source files from the vector index."""
        try:
            if self.pipeline and self.pipeline.index:
                nodes = self.pipeline.index.docstore.docs.values()
                sources = set()
                for node in nodes:
                    if hasattr(node, 'metadata') and 'source' in node.metadata:
                        sources.add(node.metadata['source'])
                return sorted(list(sources))
        except Exception as e:
            print(f"Warning: Could not retrieve sources: {e}")
        return []
    
    def chat(self, message, history, topic_filter, source_filter):
        """
        Handle chat messages with streaming support.
        
        Args:
            message: User's question
            history: Chat history
            topic_filter: Selected topic filter
            source_filter: Selected source filter
            
        Yields:
            Updated history for streaming effect
        """
        # Check if pipeline is initialized
        if self.pipeline is None:
            error_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "⚠️ Please initialize the system first by entering your API key and clicking 'Initialize System' above."}
            ]
            yield error_history
            return
        
        if not message.strip():
            yield history
            return
        
        # Add user message
        history.append({"role": "user", "content": message})
        yield history
        
        # Show processing indicator with timer and spinning icon
        start_time = time.time()
        history.append({"role": "assistant", "content": '<span class="spinning">🔄</span> Processing your query...'})
        yield history
        
        try:
            # Build filters
            filters = {}
            if topic_filter and topic_filter != "All Topics":
                filters['topic'] = topic_filter.lower()
            if source_filter and source_filter != "All Sources":
                filters['source'] = source_filter
            
            # Get answer from RAG pipeline with sources
            result = self.pipeline.query(
                message, 
                filters=filters if filters else None, 
                return_sources=True
            )
            
            # Extract response and sources
            if isinstance(result, dict):
                response = result.get('answer', str(result))
                sources = result.get('sources', [])
            else:
                response = str(result)
                sources = []
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Replace processing message and stream the response
            history[-1]["content"] = ""
            for i in range(0, len(response), 3):  # Stream 3 chars at a time for smoothness
                history[-1]["content"] = response[:i+3]
                yield history
            
            # Build sources section
            sources_text = ""
            if sources:
                sources_text = "\n\n---\n📝 **Here are the sources I used to answer your question:**\n"
                for idx, source in enumerate(sources[:3], 1):  # Show top 3 sources
                    # Handle NodeWithScore objects
                    if hasattr(source, 'node'):
                        node = source.node
                        score = source.score if hasattr(source, 'score') else None
                        # Get metadata
                        if hasattr(node, 'metadata'):
                            source_name = node.metadata.get('source', 'Unknown')
                        else:
                            source_name = 'Unknown'
                        sources_text += f"🔗 {source_name}"
                        if score:
                            sources_text += f", relevance: {score:.2f}"
                        sources_text += "\n"
                    # Handle dict format from generator
                    elif isinstance(source, dict):
                        source_name = source.get('source', f"Chunk {source.get('chunk_id', idx)}")
                        score = source.get('score')
                        sources_text += f"🔗 {source_name}"
                        if score:
                            sources_text += f", relevance: {score:.2f}"
                        sources_text += "\n"
            
            # Ensure full response is shown with timing and sources
            history[-1]["content"] = f"{response}{sources_text}\n\n_⏱️ Response time: {elapsed_time:.2f}s_"
            yield history
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
            history.append({"role": "assistant", "content": error_msg})
            yield history
    
    def create_interface(self):
        """Create and return the Gradio interface."""
        
        with gr.Blocks(title="PowerGrid AI Tutor") as interface:
            
            # Add custom CSS for spinning animation
            gr.HTML("""
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .spinning {
                    display: inline-block;
                    animation: spin 1s linear infinite;
                }
            </style>
            """)
            
            # Header
            gr.HTML("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;">
                    <h1 style="margin: 0; font-size: 2.5em;">⚡ PowerGrid AI Tutor</h1>
                    <p style="margin: 10px 0 0 0; font-size: 1.2em;">Advanced RAG System for Electrical Engineering & Renewable Energy</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">50 Research Papers | 2,166 Chunks | State-of-the-Art Retrieval</p>
                </div>
            """)
            
            # Configuration Section
            with gr.Accordion("🔧 System Configuration (Start Here!)", open=True):
                gr.Markdown("""
                ### Step 1: Configure Your System
                
                **API Key Options:**
                - **Google Gemini** (Recommended): Fast responses (~2-3s), low cost (~$0.003/query)
                  - Get free API key: [https://makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)
                - **Ollama** (Free): Slower (~30-40s), but completely free - runs locally
                  - Install Ollama first: [https://ollama.ai](https://ollama.ai)
                  - Then run: `ollama pull qwen2.5:7b`
                """)
                
                with gr.Row():
                    with gr.Column(scale=2):
                        api_key_input = gr.Textbox(
                            label="Google Gemini API Key",
                            placeholder="Paste your API key here (starts with 'AI...')",
                            type="password",
                            info="Required if using Gemini. Not needed for Ollama."
                        )
                    with gr.Column(scale=1):
                        llm_provider = gr.Radio(
                            choices=["gemini", "ollama"],
                            value="gemini",
                            label="LLM Provider",
                            info="Gemini: fast, low cost | Ollama: free, slower"
                        )
                
                gr.Markdown("### Step 2: Select RAG Features (Optional)")
                
                with gr.Row():
                    use_query_expansion = gr.Checkbox(
                        label="Query Expansion",
                        value=True,
                        info="LLM generates technical synonyms (+10-20% accuracy)"
                    )
                    use_hybrid = gr.Checkbox(
                        label="Hybrid Search",
                        value=True,
                        info="BM25 + Semantic search (+5-15% accuracy)"
                    )
                    use_reranking = gr.Checkbox(
                        label="LLM Reranking",
                        value=False,
                        info="Reorder results by relevance (+15-25% accuracy, slower)"
                    )
                
                with gr.Row():
                    init_button = gr.Button(
                        "🚀 Initialize System",
                        variant="primary",
                        size="lg"
                    )
                
                status_box = gr.Markdown("ℹ️ Please configure and initialize the system to start")
            
            gr.Markdown("---")
            
            # Filters Section
            with gr.Row():
                topic_filter = gr.Dropdown(
                    choices=["All Topics", "Solar", "Wind", "Battery", "Grid", "General"],
                    value="All Topics",
                    label="🏷️ Filter by Topic",
                    scale=1
                )
                source_filter = gr.Dropdown(
                    choices=["All Sources"],  # Will be updated after initialization
                    value="All Sources",
                    label="📄 Filter by Source Paper",
                    scale=2,
                    info="Filter results to specific research papers"
                )
            
            # Chat Interface
            chatbot = gr.Chatbot(
                height=500,
                label="💬 Chat with PowerGrid AI Tutor",
                show_label=True,
                avatar_images=(None, "⚡")
            )
            
            # Input area
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about solar energy, wind power, batteries, smart grids, or power systems...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit = gr.Button("Send", scale=1, variant="primary")
            
            # Example questions
            gr.Examples(
                examples=[
                    "What are the main challenges in integrating solar power into the electrical grid?",
                    "How does wind energy affect power grid stability?",
                    "What are the latest advances in battery energy storage systems?",
                    "Explain smart grid technology and its benefits",
                    "What is the role of inverters in solar photovoltaic systems?",
                    "How do microgrids improve grid resilience?",
                    "What are the efficiency improvements in modern solar panels?",
                ],
                inputs=msg,
                label="💡 Example Questions"
            )
            
            # Clear button
            clear = gr.Button("🗑️ Clear Chat")
            
            # Footer
            with gr.Accordion("ℹ️ About This System", open=False):
                gr.Markdown("""
                ### Technology Stack
                
                **RAG Pipeline:**
                - **Vector Store**: FAISS with 2,166 chunks
                - **Embeddings**: BAAI/bge-small-en-v1.5 (local, free)
                - **Chunk Strategy**: 512 tokens with 50-token overlap
                - **Knowledge Base**: 50 ArXiv papers on electrical engineering & renewable energy
                
                **Advanced Features:**
                1. **Query Expansion**: LLM generates related technical terms before search
                2. **Hybrid Search**: Combines BM25 (keyword) + semantic search with RRF fusion
                3. **LLM Reranking**: Re-scores retrieved chunks for better relevance
                4. **Metadata Filtering**: Filter by topic or source paper
                5. **Streaming Responses**: Real-time answer generation
                
                **Evaluation Results:**
                - Hit Rate: 50% (baseline) → 70% (full pipeline)
                - MRR: 33.9% (baseline) → 55% (full pipeline)
                - Accuracy Improvement: +30-50% with all features enabled
                
                **Cost Estimation (with Gemini):**
                - Average query: ~2,500 tokens
                - Cost per query: ~$0.003
                - Try all features for < $0.10 total
                
                **Built for**: LLM Developer Certification - Advanced RAG Project
                
                **Repository**: [GitHub](https://github.com/sudhirshivaram/powergrid-ai-tutor)
                """)
            
            # Event Handlers
            
            def update_sources_dropdown():
                """Update source filter dropdown after initialization."""
                if self.available_sources:
                    return gr.Dropdown(choices=["All Sources"] + self.available_sources[:20])
                return gr.Dropdown(choices=["All Sources"])
            
            # Initialize system
            init_button.click(
                fn=self.initialize_pipeline,
                inputs=[api_key_input, llm_provider, use_reranking, use_hybrid, use_query_expansion],
                outputs=[status_box]
            ).then(
                fn=update_sources_dropdown,
                outputs=[source_filter]
            )
            
            # Handle chat
            submit_event = submit.click(
                fn=self.chat,
                inputs=[msg, chatbot, topic_filter, source_filter],
                outputs=[chatbot]
            )
            submit_event.then(
                fn=lambda: None,
                outputs=[msg]
            )
            
            msg_event = msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot, topic_filter, source_filter],
                outputs=[chatbot]
            )
            msg_event.then(
                fn=lambda: None,
                outputs=[msg]
            )
            
            # Clear chat
            clear.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
        
        return interface
    
    def launch(self, share=False, server_name="0.0.0.0", server_port=7860):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        interface.queue()  # Enable queueing for streaming
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PowerGrid AI Tutor - With API Key Support")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    args = parser.parse_args()

    # Launch UI
    tutor = PowerGridTutorUI()
    tutor.launch(share=args.share, server_port=args.port)
