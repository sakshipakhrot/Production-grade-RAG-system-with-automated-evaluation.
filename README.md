# 🚀 Production-Grade RAG System with Automated Evaluation

A modular, scalable Retrieval-Augmented Generation (RAG) system powered by **Pinecone (vector database)** and **Groq (LLM inference)** with automated response evaluation.

---

## 🔹 Overview

This project implements an end-to-end RAG pipeline:

- Document ingestion & intelligent chunking  
- Embedding generation & storage in Pinecone  
- Semantic retrieval (Top-K similarity search)  
- Context-grounded answer generation using Groq LLM  
- Automated evaluation with structured CSV logging  

---
## 📂 Project Structure


.
├── app.py # Application entry point
├── rag_pipeline.py # Core RAG pipeline logic
├── eval.py # Automated evaluation script
├── evaluation.csv # Evaluation results & metrics
├── requirements.txt
└── README.md
