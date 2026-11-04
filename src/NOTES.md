# Credit Union Knowledge System

## Approach

This system uses a **Retrieval-Augmented Generation (RAG)** architecture with authority-based ranking to answer credit union member questions accurately from website snapshots.

### How it works

1. **Ingest**: Each snapshot's pages are chunked by markdown headings (keeping tables intact with their headers) and indexed using BM25 for fast keyword retrieval. Pages are classified into authority tiers based on URL patterns — product/rate pages are most authoritative, followed by general pages, calculators, blog posts, and newsroom articles.

2. **Answer**: For each question, BM25 retrieves the top candidate chunks. Results are re-ranked by authority tier (authoritative rate pages beat calculators and blog posts). The top context is sent to Gemini Flash, which generates a concise, specific answer. Answers are cached for consistency.

3. **Temporal queries**: Questions with an `as_of` date are answered using only pages from snapshots taken on or before that date, preferring the most recent valid snapshot.

4. **Conflicts**: The system detects rate changes across snapshots (same URL, different values over time) and cross-page contradictions (different URLs showing different rates in the same snapshot). Conflicts are resolved using authority ranking — the most authoritative, most recent source wins.

### Key design decisions

- **Authority hierarchy is hard-coded**: Product/rate pages always beat calculators and blog posts. This prevents stale calculator values from appearing in answers.
- **Answer caching**: Every (question, as_of) pair is cached after the first call, ensuring perfect consistency on repeat questions.
- **Graceful refusal**: When the website doesn't contain the answer, the system says so honestly rather than guessing.

## Dependencies

- `google-genai` — Gemini Flash API for answer generation

## API Keys

This system requires a **Google Gemini API key** set as the `GEMINI_API_KEY` environment variable. It uses Gemini 3.8 Flash for answer generation. Without the key, the system falls back to a rule-based extraction method with lower accuracy.
