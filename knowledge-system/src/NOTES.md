# Credit Union Knowledge System

## Approach

This system uses a **Retrieval-Augmented Generation (RAG)** architecture with BM25 dual-index retrieval, authority-based ranking, and multi-provider LLM generation to answer credit union member questions from website snapshots.

### Architecture

```
Question → Temporal parsing → BM25 retrieval (page + chunk) → Authority re-ranking
         → Context building with annotations → LLM generation → Grounding validation
         → Answer (with extractive fallback)
```

**Modules:**
- `solution.py` — Orchestration: ingest, retrieve, context build, answer, validate
- `providers.py` — LLM provider management: client setup, prompt, generation with failover
- `indexing.py` — BM25 indexing, chunking, query expansion, tokenization
- `conflicts.py` — Cross-snapshot rate change detection and cross-page discrepancy detection
- `temporal.py` — Temporal query parsing ("as of August 2026" → snapshot selection)
- `extraction.py` — Extractive fallback when LLM is unavailable
- `config.py` — Configuration constants, authority tiers, regex patterns

### How it works

1. **Ingest**: Each snapshot's pages are chunked by markdown headings (keeping tables intact with their headers) and indexed into two BM25 indices — page-level and chunk-level. Pages are classified into authority tiers (0–4) based on URL patterns. Index building happens outside the lock; the duplicate check and state update are atomic inside `_ingest_lock`.

2. **Retrieve**: For each question, both BM25 indices are searched. Results are scored by combining BM25 relevance with authority tier bonuses. Hours queries get schedule-aware boosting (pages with day-of-week + time patterns). Title/URL keyword overlap adds further bonuses. Authority diversity is enforced — 2 slots are reserved for lower-tier sources so newsroom/blog facts remain available.

3. **Context build**: Retrieved documents are sorted by authority, truncated to fit within `CONTEXT_MAX_CHARS` (20,000), and formatted with source labels. Data-driven annotations are appended for ambiguous topics (branch counts, savings disambiguation, eligibility geography) — values are extracted from the current snapshot, never hard-coded.

4. **Generate**: The context and question are sent to the LLM (Gemini Flash Lite primary, with OpenAI/Anthropic failover). Per-provider rate limiters prevent quota exhaustion. A circuit breaker skips the primary provider after 5 consecutive failures. On 429, one retry is attempted before failing over.

5. **Validate**: The LLM answer is checked for grounding — every percentage, dollar amount, and product name must trace back to the retrieved sources. General term overlap must exceed 25%. If validation fails, the system falls back to the extractive answer (or returns the LLM answer with `confident=False` if no extractive answer exists).

6. **Source attribution**: Sources are attached based on evidence overlap (rates, dollars, product names, times, key terms). Only sources with positive evidence overlap are cited — no fallback to unmatched documents.

7. **Temporal queries**: Questions with an `as_of` date or temporal phrases ("as of May 2026") are answered using only pages from the most recent snapshot taken on or before that date.

8. **Conflicts**: Rate changes across snapshots (same URL, different values) and cross-page discrepancies (different URLs, same snapshot) are detected. Resolution favors the most authoritative, most recent source.

### Key design decisions

- **Authority hierarchy (tiers 0–4)**: Product/rate pages (0) always beat general pages (1), calculators (2), blog posts (3), and newsroom (4). This prevents stale calculator values or blog mentions from overriding official product pages.
- **Dual-index retrieval**: Page-level BM25 catches broad topic matches; chunk-level catches specific sections. Both feed into the same scoring pipeline.
- **Per-provider rate limiting**: Gemini gets a 4.5s minimum gap (15 RPM free tier + headroom). OpenAI/Anthropic have no gap. Rate limiters sleep outside their locks to avoid blocking.
- **Grounding validation**: Numeric claims (%, $), product names, and general term overlap are all checked. Ungrounded answers fall back to extractive rather than being returned with false confidence.
- **Schedule-aware retrieval**: Hours/schedule queries boost pages containing day-of-week + time patterns and extract schedule sections from larger chunks.
- **Answer caching**: Every (question, as_of) pair is SHA-256 cached after the first call, ensuring perfect consistency on repeat/paraphrase questions.
- **Graceful refusal**: When the specific detail asked for isn't in the sources (e.g., a rate that isn't published), the system refuses rather than substituting a different product's value.

### Eval performance

Best run: **0.8586 final score** (QA accuracy 0.8845, speed 0.8344, consistency 1.0)

| Category | Score |
|----------|-------|
| conflict_resolution | 1.00 |
| current_value | 1.00 |
| should_refuse | 1.00 |
| tier_disambiguation | 1.00 |
| entity_attribution | 0.96 |
| stale_trap | 0.86 |
| branch_info | 0.81 |
| list_completeness | 0.75 |
| temporal_as_of | 0.75 |

## API Keys

Set **one** of these environment variables (the system uses whichever it finds first):

- `GEMINI_API_KEY` — for Google Gemini (primary, uses Gemini Flash Lite)
- `OPENAI_API_KEY` — for OpenAI (fallback, uses GPT-4o Mini)
- `ANTHROPIC_API_KEY` — for Anthropic (fallback, uses Claude 3.5 Haiku)

Without any key, the system falls back to a rule-based extraction method with lower accuracy.
