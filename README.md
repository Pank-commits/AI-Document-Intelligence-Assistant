# AI Document Intelligence Assistant

AI-powered Retrieval-Augmented Generation (RAG) system for intelligent multi-document question answering using semantic search, Qdrant vector database, and LLM APIs.

---

## Features

- Multi-document question answering
- Semantic search using vector embeddings
- Hybrid retrieval using Qdrant + keyword search
- Citation-based response generation
- Structured and unstructured query handling
- Low-latency document retrieval
- Interactive user interface

---

## Tech Stack

- Python
- Qdrant
- NLP
- LLM APIs
- Docker
- Streamlit
- Semantic Search

---

## Project Architecture

```text
User Query
   ↓
Query Processing
   ↓
Semantic Retrieval (Qdrant)
   ↓
Hybrid Search
   ↓
LLM Response Generation
   ↓
Citation Builder
   ↓
Final Answer
