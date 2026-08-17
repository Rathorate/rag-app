# Talk to My Documents

A personal RAG (Retrieval-Augmented Generation) app — upload your own PDFs and ask questions about them. Answers are grounded in your actual documents instead of the model's general training knowledge, with source citations showing which part of the document each answer came from.

## Status

🚧 In development — currently in the setup/ingestion phase. See [Roadmap](#roadmap) below.

## How it works

```
Your PDF(s)
    ↓
Load & Chunk        → split documents into small, searchable pieces
    ↓
Embed                → turn each chunk into a vector (numeric representation of meaning)
    ↓
Store                → save vectors in a local vector database (Chroma)

[User asks a question]
    ↓
Embed the question   → same embedding model, same vector space
    ↓
Retrieve             → find the chunks whose vectors are closest to the question
    ↓
Generate             → send question + retrieved chunks to the LLM, answer using only that context
    ↓
Display              → show the answer plus which source chunk/page it came from
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python | Simple ecosystem for this kind of app |
| LLM + Embeddings | Google Gemini (free tier) | One API key covers both chat and embeddings |
| Vector store | ChromaDB | Runs locally, no signup, no cost |
| PDF parsing | pypdf | Lightweight, no external service needed |
| UI | Streamlit | Fast working chat interface, no frontend code needed |

## Features

- [x] Project scaffolding and framework defined
- [ ] Load and chunk PDF documents
- [ ] Generate embeddings and store them in ChromaDB
- [ ] Retrieve relevant chunks for a given question
- [ ] Generate grounded answers with the LLM (refuses to answer outside the document)
- [ ] Source citations shown under each answer
- [ ] Streamlit chat interface
- [ ] Multi-document support
- [ ] Handles messy/real-world PDFs (scanned pages, multi-column layouts)

## Getting started

### Prerequisites

- Python 3.10+
- A free [Gemini API key](https://ai.google.dev)

### Setup

```bash
# Clone or navigate into the project folder
cd rag-app

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install streamlit google-generativeai chromadb pypdf

# Set your API key
export GEMINI_API_KEY="your-key-here"     # Mac/Linux
set GEMINI_API_KEY="your-key-here"        # Windows
```

### Running the app

```bash
streamlit run app.py
```

*(This will work once `app.py` exists — currently still in the ingestion-script stage. See Roadmap.)*

## Project structure

```
rag-app/
├── app.py              # Streamlit UI (not yet built)
├── ingest.py           # PDF loading + chunking (in progress)
├── embed.py            # Embedding + ChromaDB storage (not yet built)
├── query.py            # Retrieval + generation logic (not yet built)
├── requirements.txt
└── README.md
```

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 1. Ingestion | Load and chunk PDFs | 🚧 In progress |
| 2. Embedding + storage | Generate and store vector embeddings | ⬜ Not started |
| 3. Retrieval + generation | Answer questions grounded in retrieved chunks | ⬜ Not started |
| 4. UI | Streamlit chat interface with citations | ⬜ Not started |
| 5. Polish | Stress-test with real-world PDFs, tune chunking | ⬜ Not started |
| 6. Stretch goals | Multi-file-type support, persistent memory, deployment | ⬜ Optional |

Estimated timeline: ~3 weeks part-time.

## Design principles

- **Don't hallucinate.** If the answer isn't in the retrieved chunks, the app should say so rather than falling back on the model's general knowledge.
- **Show your work.** Every answer should be traceable to a specific chunk/page in the source document.
- **Test each layer alone.** Ingestion, embedding, retrieval, and generation are each verified independently before being wired together end-to-end.

## License

Personal project — no license specified yet.
