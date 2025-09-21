# RAG Demo Project

## Overview

This project is a Retrieval-Augmented Generation (RAG) system built with Node.js, LangChain, and OpenAI. It answers guest-facing questions such as hours, ticketing, and parking using content from your own documents.

## Features

* Ingestion pipeline: load, chunk, embed, and snapshot documents into `index.json`.
* Semantic retrieval with distance gating for confidence control.
* Guardrails for vague queries and synonym expansion.
* Clear citations: only show sources actually used by the model.
* Minimal HTML/CSS UI with a clean, modern theme.

## How It Works

1. **Ingest**: `npm run ingest` reads `data/*.txt`, splits into chunks, embeds them, and saves to `index.json`.
2. **Serve**: `npm run serve` starts an Express server with `/ask` endpoint.
3. **Retrieve**: Top K document chunks are retrieved using similarity search and distance thresholding.
4. **Generate**: LLM produces an answer using only the retrieved context and cites the sources.

## Stack

* Node.js + Express (server)
* LangChain (orchestration)
* OpenAI (embeddings and LLM)
* Embeddings: text-embedding-3-small
* LLM: gpt-4o-mini (OpenAI)
* Vector store: LangChain MemoryVectorStore (local demo)
* HTML/CSS frontend

## Architecture

User (browser)
   ↓  POST /ask
Express API (src/server.mjs)
   ├─ expand synonyms / validate input
   ├─ retrieve top-K with scores (vector store)
   ├─ distance gate (confidence check)
   ├─ build numbered context blocks [1], [2], ...
   ├─ LLM generate (grounded; short; cited)
   └─ return { answer, sources }

Ingestion (src/ingest.mjs)
   ├─ read data/*.txt
   ├─ paragraph chunking
   ├─ compute embeddings (OpenAI)
   └─ snapshot as index.json (docs + metadata)

## Running the Project

```bash
npm install
npm run ingest
npm run serve
```

Open `http://localhost:3000` in your browser.

## Scaling Roadmap

* Replace MemoryVectorStore with Pinecone or Chroma.
* Add hybrid search (keyword + embeddings).
* Introduce re-ranking and MMR for improved retrieval.
* Expand guardrails for synonyms and vague queries.

## License

MIT
