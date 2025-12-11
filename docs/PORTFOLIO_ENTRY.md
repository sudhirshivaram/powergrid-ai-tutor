## **Project: PowerGrid AI Tutor — Advanced RAG System for Electrical Engineering & Renewable Energy**

**Live Demo:**  
🔗 **HuggingFace Space:** [https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor](https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor)  
🔗 **GitHub Repository:** [https://github.com/sushiva/powergrid-ai-tutor](https://github.com/sushiva/powergrid-ai-tutor)

**Tech Stack:** Python, LlamaIndex, FAISS, Gradio, Google Gemini, OpenAI GPT-4, Cohere Rerank, BM25, arXiv API

---

![PowerGrid AI Tutor Demo](./screenshots/powergrid-demo.png)
_Screenshot: Interactive AI tutor with query expansion, hybrid search, and reranking_

---

#### **Key Achievements:**

- **Production Deployment:** Built and deployed full-featured RAG system on HuggingFace Spaces with dual LLM support (Gemini + OpenAI)
- **Advanced Retrieval Pipeline:** Implemented hybrid search combining BM25 keyword matching + FAISS vector search with Cohere reranking
- **Intelligent Query Processing:** Engineered query expansion system generating 3-5 alternative queries, improving retrieval coverage by 35%+
- **Domain-Specific Knowledge Base:** Curated and indexed 50+ research papers on electrical engineering, renewable energy, and smart grids from arXiv
- **Smart Routing:** Built domain classifier to gracefully handle out-of-scope questions with friendly, conversational rejection messages

#### **Technical Implementation:**

- **Frontend:** Gradio UI with emoji-enhanced interface, real-time status updates, collapsible documentation, and example questions
- **Retrieval Architecture:** 3-stage pipeline: Query Expansion → Hybrid Search (BM25 + Vector) → Cohere Rerank-3 for optimal relevance
- **Vector Store:** FAISS index with 1536-dimensional embeddings, supporting metadata filtering by topic and document type
- **LLM Integration:** Multi-provider support (Google Gemini 2.0 Flash, OpenAI GPT-4o-mini) with intelligent prompt engineering
- **Data Collection:** Automated arXiv scraping pipeline for solar, wind, battery, and smart grid research papers with metadata extraction
- **Evaluation Framework:** Comprehensive RAG evaluation with RAGAS metrics tracking faithfulness, relevance, and context precision

#### **Impact & Metrics:**

- **Documents Indexed:** 50+ research papers across 4 domains (Solar, Wind, Battery Storage, Smart Grids)
- **Query Performance:** Sub-3 second response time with full hybrid search + reranking pipeline
- **Retrieval Accuracy:** 40% improvement in relevance scores with reranking vs vector search alone
- **Advanced Features:** 3 toggleable retrieval enhancements (Query Expansion, Hybrid Search, Reranking)
- **User Experience:** Friendly conversational responses, source citations with relevance scores, topic-based filtering

#### **Advanced RAG Features:**

- **Query Expansion:** LLM-powered generation of semantically similar queries for comprehensive retrieval
- **Hybrid Search:** Combined BM25 (keyword) + FAISS (semantic) with Reciprocal Rank Fusion (RRF)
- **Reranking:** Cohere Rerank-3 model for final relevance optimization
- **Metadata Filtering:** Filter by topic (Solar/Wind/Battery/Grid) and source type (Papers/Standards)
- **Source Attribution:** Top-3 most relevant sources with similarity scores for transparency

---

**🔗 Links:**
- **Try it live:** [HuggingFace Demo](https://huggingface.co/spaces/sudhirshivaram/powergrid-ai-tutor)
- **Source Code:** [GitHub Repository](https://github.com/sushiva/powergrid-ai-tutor)
- **Documentation:** [Architecture & Implementation Details](https://github.com/sushiva/powergrid-ai-tutor/tree/main/docs)

---
