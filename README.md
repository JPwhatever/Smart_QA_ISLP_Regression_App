# ISLP Linear Regression QA System

A Retrieval-Augmented Generation (RAG) system for answering questions about linear regression from the ISLP textbook.

## Architecture


## Components

- **PDF Parsing**: Marker (ML-based, preserves LaTeX equations)
- **Chunking**: Section-based, 28 chunks from ISLP Chapter 3
- **Dense Retrieval**: BGE (bge-base-en-v1.5) — contrastive retrieval training
- **Sparse Retrieval**: BM25 — keyword matching with saturation and length normalization
- **Fusion**: Reciprocal Rank Fusion (RRF) — rank-based combination
- **Generation**: Gemini 2.5 Flash (primary), Llama 3.1 8B via Groq (comparison)

## Evaluation Results

### Full Eval Set (25 questions)

| Method | Hit@1 | Hit@3 | MRR |
|--------|-------|-------|-----|
| BGE alone | 0.48 | 0.68 | 0.567 |
| SBERT alone | 0.40 | 0.68 | 0.533 |
| BM25 alone | 0.60 | 0.80 | 0.687 |
| BGE + BM25 | 0.52 | 0.76 | 0.633 |
| SBERT + BM25 | 0.52 | 0.80 | 0.653 |

### Key Findings

- BM25 outperforms dense embeddings on keyword questions due to precise technical terminology
- Dense embeddings show superior Hit@3 on semantic questions (0.80 vs 0.60 for BM25)
- Hybrid retrieval is most robust across mixed query types
- Gemini 2.5 Flash significantly outperforms Llama 3.1 8B on ambiguous questions

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add API keys to `.streamlit/secrets.toml`:

## Tech Stack

- BGE embeddings (BAAI/bge-base-en-v1.5)
- BM25 (rank-bm25)
- RRF fusion
- Gemini 2.5 Flash
- Llama 3.1 8B via Groq
- Streamlit
