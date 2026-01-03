---
title: "Contextual RAG: Adding Context to Chunks for Better Retrieval"
date: 2025-12-31
description: "Improve your RAG pipeline by prepending contextual information to each chunk before embedding, reducing retrieval failures by up to 67%."
tags: [AI, RAG, Retrieval, Python, Tutorial]
---

Standard RAG has a fundamental problem: chunks lose their context. When you split a document into pieces, each chunk becomes isolated. A sentence like "The company increased revenue by 15%" is meaningless without knowing which company, which year, and what the previous revenue was.

Contextual RAG solves this by enriching each chunk with surrounding context before embedding. Anthropic's research shows this technique can reduce retrieval failures by up to 67%.

## The Problem with Standard Chunking

Consider this document:

```
# Q3 2025 Financial Report

## Executive Summary
Revenue grew 15% year-over-year to $4.2 billion.

## Regional Performance
### North America
The region exceeded targets with $2.1 billion in sales.

### Europe
Growth slowed to 8% due to currency headwinds.
```

Standard chunking might create:

```
Chunk 1: "Revenue grew 15% year-over-year to $4.2 billion."
Chunk 2: "The region exceeded targets with $2.1 billion in sales."
Chunk 3: "Growth slowed to 8% due to currency headwinds."
```

Now if a user asks "How did the European region perform?", Chunk 3 might not be retrieved because it doesn't mention "Europe"—that context is in the header above.

## Contextual RAG Solution

Before embedding, we prepend context to each chunk:

```
Chunk 1: "This is from the Executive Summary section of the Q3 2025 Financial Report. Revenue grew 15% year-over-year to $4.2 billion."

Chunk 2: "This is from the North America subsection of Regional Performance in the Q3 2025 Financial Report. The region exceeded targets with $2.1 billion in sales."

Chunk 3: "This is from the Europe subsection of Regional Performance in the Q3 2025 Financial Report. Growth slowed to 8% due to currency headwinds."
```

Now "Europe" appears in Chunk 3, and retrieval succeeds.

## Implementation

### Step 1: Generate Context with an LLM

Use a language model to generate a brief context description for each chunk:

```python
from openai import OpenAI

client = OpenAI()

def generate_context(document: str, chunk: str) -> str:
    """Generate contextual description for a chunk."""
    prompt = f"""
    <document>
    {document[:3000]}  # First part of document for context
    </document>

    Here is a chunk from the document:
    <chunk>
    {chunk}
    </chunk>

    Provide a brief (1-2 sentence) description of what this chunk is about
    and where it fits in the document. This will be prepended to the chunk
    for better search retrieval.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )

    return response.choices[0].message.content.strip()
```

### Step 2: Contextual Chunking Pipeline

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_contextual_chunks(document: str, chunk_size: int = 500) -> list[dict]:
    """Split document and add context to each chunk."""

    # Standard chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    raw_chunks = splitter.split_text(document)

    contextual_chunks = []
    for i, chunk in enumerate(raw_chunks):
        # Generate context
        context = generate_context(document, chunk)

        # Create contextualized chunk
        contextual_chunk = f"{context}\n\n{chunk}"

        contextual_chunks.append({
            "id": f"chunk_{i}",
            "original": chunk,
            "context": context,
            "contextual_text": contextual_chunk
        })

    return contextual_chunks
```

### Step 3: Embed and Store

Embed the contextualized text, not the original:

```python
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("contextual_docs")

def index_contextual_chunks(chunks: list[dict]):
    """Index contextualized chunks in vector store."""

    for chunk in chunks:
        # Embed the contextual text
        embedding = embedder.encode(chunk["contextual_text"]).tolist()

        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["original"]],  # Store original for display
            metadatas=[{"context": chunk["context"]}]
        )
```

### Step 4: Retrieval

At query time, you can optionally add context to the query too:

```python
def contextual_search(query: str, n_results: int = 5) -> list[dict]:
    """Search with optional query contextualization."""

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return [
        {
            "text": doc,
            "context": meta["context"]
        }
        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        )
    ]
```

## Optimization: Caching Context Generation

Generating context for every chunk is expensive. Cache the results:

```python
import hashlib
import json
from pathlib import Path

CACHE_DIR = Path("./context_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_context(document: str, chunk: str) -> str | None:
    """Check cache for existing context."""
    cache_key = hashlib.md5(f"{document[:500]}{chunk}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text())["context"]
    return None

def save_context_to_cache(document: str, chunk: str, context: str):
    """Save generated context to cache."""
    cache_key = hashlib.md5(f"{document[:500]}{chunk}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"

    cache_file.write_text(json.dumps({"context": context}))

def generate_context_with_cache(document: str, chunk: str) -> str:
    """Generate context with caching."""
    cached = get_cached_context(document, chunk)
    if cached:
        return cached

    context = generate_context(document, chunk)
    save_context_to_cache(document, chunk, context)
    return context
```

## Combining with BM25

For even better retrieval, combine contextual embeddings with BM25:

```python
import numpy as np
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks

        # BM25 on contextual text
        tokenized = [c["contextual_text"].split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        # Vector store
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = self.embedder.encode(
            [c["contextual_text"] for c in chunks]
        )

    def search(self, query: str, n: int = 5, alpha: float = 0.5) -> list[dict]:
        """Hybrid search with weighted combination."""
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_scores = bm25_scores / bm25_scores.max()  # Normalize

        # Vector scores
        query_emb = self.embedder.encode(query)
        vector_scores = np.dot(self.embeddings, query_emb)
        vector_scores = (vector_scores - vector_scores.min()) / (
            vector_scores.max() - vector_scores.min()
        )

        # Combine scores
        combined = alpha * vector_scores + (1 - alpha) * bm25_scores

        # Get top n
        top_indices = np.argsort(combined)[-n:][::-1]

        return [self.chunks[i] for i in top_indices]
```

## When to Use Contextual RAG

**Use it when:**
- Documents have hierarchical structure (headers, sections)
- Chunks reference entities defined elsewhere
- You're seeing retrieval failures on relevant content
- Document length makes whole-document embedding impractical

**Skip it when:**
- Documents are short and self-contained
- Each chunk is naturally independent (e.g., FAQ entries)
- Cost of context generation is prohibitive

## Cost Analysis

Contextual RAG adds LLM costs at indexing time:

| Documents | Chunks | Context Calls | Cost (GPT-4o-mini) |
|-----------|--------|---------------|-------------------|
| 100 | 1,000 | 1,000 | ~$0.15 |
| 1,000 | 10,000 | 10,000 | ~$1.50 |
| 10,000 | 100,000 | 100,000 | ~$15.00 |

This is a one-time indexing cost. Query costs remain the same.

## Measuring Improvement

Track these metrics before and after:

```python
def evaluate_retrieval(queries: list[str], ground_truth: list[list[str]]):
    """Evaluate retrieval quality."""
    hits = 0
    total = 0

    for query, expected_chunks in zip(queries, ground_truth):
        results = contextual_search(query, n_results=5)
        retrieved_ids = [r["id"] for r in results]

        for expected in expected_chunks:
            total += 1
            if expected in retrieved_ids:
                hits += 1

    recall = hits / total
    return recall
```

Anthropic's benchmarks show:
- **Standard RAG recall:** ~60%
- **Contextual RAG recall:** ~84%
- **Contextual + Hybrid:** ~95%

## Complete Example

```python
# Full pipeline
document = open("quarterly_report.md").read()

# 1. Create contextual chunks
chunks = create_contextual_chunks(document)

# 2. Index
index_contextual_chunks(chunks)

# 3. Query
results = contextual_search("What was the European performance?")

for r in results:
    print(f"Context: {r['context']}")
    print(f"Content: {r['text'][:200]}...")
    print("---")
```

## What's Next

Contextual RAG is one technique in the Advanced RAG toolkit. Other improvements to explore:

*   **Query rewriting:** Transform user queries before retrieval
*   **Reranking:** Use a cross-encoder to rerank retrieved results
*   **Document hierarchies:** Maintain parent-child relationships between chunks
*   **Late chunking:** Embed full documents, then chunk the embeddings

The key insight: RAG quality is mostly about retrieval quality. Invest in making your chunks as searchable as possible, and everything downstream improves.

---

## Try It Yourself

Copy this prompt into your AI coding agent to build this project:

```
Build a Contextual RAG pipeline that improves retrieval by adding context:
1. A context generator that uses an LLM to describe each chunk's position
2. A chunking pipeline that prepends context to chunks before embedding
3. A ChromaDB vector store that indexes contextualized text
4. A hybrid retriever combining BM25 and vector search with alpha weighting

Include caching for context generation using MD5 hashes. Test with a
hierarchical document (with headers/sections) and show how contextual
chunks improve retrieval for queries referencing section context.
```
