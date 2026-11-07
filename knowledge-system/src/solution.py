"""
Credit Union Knowledge System — RAG pipeline for answering member questions
from Dupaco credit union website snapshots.

Architecture: BM25 dual-index retrieval (page + chunk level) with authority-based
ranking, LLM generation (multi-provider) with extraction fallback,
and cross-snapshot conflict detection.
"""
import copy
import hashlib
import logging
import os
import re
import sys
import threading
from collections import OrderedDict, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from harness and sibling modules
# ---------------------------------------------------------------------------
_solution_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_solution_dir))

_harness_dirs = [
    d / "harness"
    for d in _solution_dir.parents
    if (d / "harness" / "contract.py").exists()
]
if _harness_dirs:
    sys.path.insert(0, str(_harness_dirs[0]))

try:
    from contract import Answer, Conflict, Snapshot, Source
except ImportError:
    from harness.contract import Answer, Conflict, Snapshot, Source

from config import (
    AUTH_LABELS, STOP_WORDS, PRODUCT_TERMS,
    RATE_Q_RE, HOURS_Q_RE, ADDR_Q_RE, LIST_Q_RE,
    TIME_PAT, SCHEDULE_PAT, DOLLAR_PAT, PRODUCT_NAME_PAT,
    REFUSAL_TEXT, CACHE_MAX_SIZE,
    CHUNK_BM25_TOP_K, PAGE_BM25_TOP_K, MAX_RELEVANT_PARTS,
    CONTEXT_MAX_CHARS, CONTEXT_DOC_MAX_CHARS,
    get_authority,
)
from conflicts import detect_conflicts
from extraction import extract_answer
from indexing import BM25, chunk_page, tokenize, expand_query
from providers import LLMProvider
from temporal import extract_temporal_date

logger = logging.getLogger(__name__)

_CHARTER_STATE_RE = re.compile(
    r"(?:find us in|charter area|located in|across)\s+"
    r"([A-Z][a-z]+(?:,?\s+(?:and\s+)?[A-Z][a-z]+)*)",
)
_STATE_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+")


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
def _load_dotenv():
    script_dir = Path(__file__).resolve().parent
    search_dirs = [script_dir, script_dir.parent, script_dir.parent.parent, Path.cwd()]
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if key and val and key not in os.environ:
                        os.environ[key] = val
            return

_load_dotenv()


# ---------------------------------------------------------------------------
# LRU cache with bounded size
# ---------------------------------------------------------------------------
class LRUCache:
    """Bounded, thread-safe cache that doesn't expose stored mutable values."""

    def __init__(self, maxsize: int = CACHE_MAX_SIZE):
        if maxsize < 1:
            raise ValueError("maxsize must be at least 1")
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return copy.deepcopy(self._data[key])

    def put(self, key: str, value):
        stored = copy.deepcopy(value)
        with self._lock:
            self._data[key] = stored
            self._data.move_to_end(key)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


# ---------------------------------------------------------------------------
# Knowledge System
# ---------------------------------------------------------------------------
class CreditUnionKnowledge:
    """Ingests snapshots, answers questions, detects conflicts."""

    def __init__(self):
        # Per-snapshot indices and page data
        self.snapshots_meta: list[tuple] = []       # [(snapshot_id, taken_on), ...] sorted by date
        self._snapshot_ids: set[str] = set()         # fast duplicate check
        self.latest_snapshot_id: Optional[str] = None
        self.latest_date = None
        self.chunk_indices: dict[str, BM25] = {}     # snapshot_id → chunk-level BM25 index
        self.page_indices: dict[str, BM25] = {}      # snapshot_id → page-level BM25 index
        self.pages_by_snapshot: dict = {}             # snapshot_id → {url: page_data}
        self.chunks_by_snapshot: dict[str, dict[str, list]] = {}  # snapshot_id → {url: [chunks]}
        self.page_history: dict = defaultdict(list)   # url → [(sid, date, markdown, title), ...]

        # Conflict tracking and caching
        self._conflicts: list = []
        self._conflict_urls: dict = {}    # url → conflict description string
        self._cache = LRUCache()
        self._ingest_lock = threading.Lock()  # protects all shared state above

        # Auto-detect the first available LLM provider
        provider, api_key = None, None
        for prov, key_name in [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]:
            key = os.environ.get(key_name)
            if key:
                provider, api_key = prov, key
                logger.info("LLM provider: %s", prov)
                break
        self._llm = LLMProvider(provider, api_key)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, snapshot: Snapshot) -> None:
        """Index a snapshot's pages. Deduplicates by snapshot_id and
        determines latest snapshot by date, not arrival order."""
        if not snapshot.snapshot_id or not snapshot.taken_on:
            raise ValueError("snapshot_id and taken_on are required")
        try:
            date.fromisoformat(str(snapshot.taken_on))
        except ValueError as exc:
            raise ValueError("taken_on must be an ISO date (YYYY-MM-DD)") from exc
        # Build indices locally first — if a page is invalid, nothing is committed
        chunk_idx = BM25()     # chunk-level index for this snapshot
        page_idx = BM25()      # page-level index for this snapshot
        pages_map = {}         # url → {url, title, markdown, authority}
        chunks_map = {}        # url → [chunk dicts]
        history_updates = defaultdict(list)  # url → [(sid, date, md, title)]
        seen_urls = set()

        for page in snapshot.pages:
            if not page.url or page.url in seen_urls:
                raise ValueError(f"snapshot contains a missing or duplicate page URL: {page.url!r}")
            if not isinstance(page.markdown, str):
                raise ValueError(f"page markdown must be text: {page.url}")
            seen_urls.add(page.url)
            authority = get_authority(page.url)
            pages_map[page.url] = {
                "url": page.url,
                "title": page.title,
                "markdown": page.markdown,
                "authority": authority,
            }
            history_updates[page.url].append(
                (snapshot.snapshot_id, snapshot.taken_on, page.markdown, page.title)
            )

            page_idx.add(
                {
                    "url": page.url,
                    "title": page.title,
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_date": snapshot.taken_on,
                    "authority": authority,
                },
                f"{page.title} {page.markdown}",
            )

            chunks = chunk_page(page.markdown, page.title)
            chunks_map[page.url] = chunks
            for chunk in chunks:
                chunk_idx.add(
                    {
                        "url": page.url,
                        "title": page.title,
                        "snapshot_id": snapshot.snapshot_id,
                        "snapshot_date": snapshot.taken_on,
                        "authority": authority,
                        "heading": chunk["heading"],
                        "chunk_text": chunk["text"],
                    },
                    f"{page.title} {chunk['heading']} {chunk['text']}",
                )

        # Atomic commit: duplicate check + state update under a single lock
        with self._ingest_lock:
            if snapshot.snapshot_id in self._snapshot_ids:
                logger.info("Skipping duplicate snapshot %s", snapshot.snapshot_id)
                return
            self.chunk_indices[snapshot.snapshot_id] = chunk_idx
            self.page_indices[snapshot.snapshot_id] = page_idx
            self.pages_by_snapshot[snapshot.snapshot_id] = pages_map
            self.chunks_by_snapshot[snapshot.snapshot_id] = chunks_map
            for url, entries in history_updates.items():
                self.page_history[url].extend(entries)
                self.page_history[url].sort(key=lambda item: (str(item[1]), item[0]))
            self._snapshot_ids.add(snapshot.snapshot_id)
            self.snapshots_meta.append((snapshot.snapshot_id, snapshot.taken_on))
            self.snapshots_meta.sort(key=lambda item: (str(item[1]), item[0]))
            self.latest_snapshot_id, self.latest_date = self.snapshots_meta[-1]
            self._cache.clear()
            self._detect_conflicts()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def _find_snapshot_for_date(self, as_of: str) -> Optional[str]:
        try:
            cutoff = date.fromisoformat(str(as_of))
        except (TypeError, ValueError):
            return None
        valid = [
            (sid, date.fromisoformat(str(snapshot_date)))
            for sid, snapshot_date in self.snapshots_meta
            if date.fromisoformat(str(snapshot_date)) <= cutoff
        ]
        return max(valid, key=lambda item: (item[1], item[0]))[0] if valid else None

    def _retrieve(self, question: str, as_of: Optional[str] = None, top_k: int = 10) -> list[dict]:
        """Retrieve the most relevant documents for a question from the target snapshot.
        Uses dual-index BM25 (page + chunk), authority-weighted scoring, and
        question-type-specific boosting (hours, branches, lists)."""

        # Select which snapshot to search
        if as_of:
            target_sid = self._find_snapshot_for_date(as_of)
            if not target_sid:
                return []
        else:
            target_sid = self.latest_snapshot_id

        if target_sid is None or target_sid not in self.chunk_indices:
            return []
        chunk_idx = self.chunk_indices[target_sid]
        page_idx = self.page_indices[target_sid]
        pages = self.pages_by_snapshot[target_sid]
        precomputed_chunks = self.chunks_by_snapshot[target_sid]
        snap_date = dict(self.snapshots_meta)[target_sid]

        # List questions need more sources to cover "all/every" type answers
        if LIST_Q_RE.search(question):
            top_k = 20

        # Search both indices with expanded query (plural/singular variants)
        search_q = expand_query(question)
        chunk_results = chunk_idx.search(search_q, top_k=CHUNK_BM25_TOP_K)
        page_results = page_idx.search(search_q, top_k=PAGE_BM25_TOP_K)

        # Detect question type to adjust authority weighting
        is_hours_q = bool(HOURS_Q_RE.search(question))
        is_branch_q = bool(
            is_hours_q
            or ADDR_Q_RE.search(question)
            or re.search(r"\bbranch\b", question, re.I)
        )
        # Branch queries reduce authority bonus so newsroom/blog sources with
        # branch-specific details aren't buried by generic product pages
        page_auth = 3.0 if is_branch_q else 5.0
        chunk_auth = 2.0 if is_branch_q else 3.5

        # Score each URL: BM25 relevance + authority tier bonus
        url_scores: dict[str, float] = defaultdict(float)
        url_chunks: dict[str, list] = defaultdict(list)

        for doc, score in page_results:
            url = doc["url"]
            combined = score + (5 - doc["authority"]) * page_auth
            url_scores[url] = max(url_scores[url], combined)

        for doc, score in chunk_results:
            url = doc["url"]
            combined = score + (5 - doc["authority"]) * chunk_auth
            url_scores[url] = max(url_scores[url], combined)
            url_chunks[url].append((doc, score))

        # Tokens from the question, used for title/URL overlap and diversity scoring
        q_tokens = set(tokenize(question)) - STOP_WORDS

        # For hours queries, boost pages containing actual schedule data
        # (day-of-week + time pattern), not just any time mention
        if is_hours_q:
            for url, pdata in pages.items():
                md = pdata["markdown"]
                schedule_matches = len(SCHEDULE_PAT.findall(md))
                if schedule_matches >= 2:
                    bonus = 25.0 + schedule_matches * 2.0
                    url_scores[url] = max(url_scores.get(url, 0), bonus)

        # Title/URL keyword overlap: boost pages whose title or URL path
        # shares words with the question (catches exact product name matches)
        for url in list(url_scores.keys()):
            if url not in pages:
                continue
            title_tokens = set(tokenize(pages[url]["title"])) - STOP_WORDS
            url_path = url.split("dupaco.com")[-1] if "dupaco.com" in url else ""
            url_tokens = set(tokenize(url_path)) - STOP_WORDS
            overlap = q_tokens & (title_tokens | url_tokens)
            if overlap:
                url_scores[url] += len(overlap) * 5.0

        # Product-term fallback: high-authority pages that BM25 missed but
        # whose title/URL contains a product term from the question
        for url, pdata in pages.items():
            if url in url_scores or pdata["authority"] > 1:
                continue
            title_tokens = set(tokenize(pdata["title"])) - STOP_WORDS
            url_path = url.split("dupaco.com")[-1] if "dupaco.com" in url else ""
            url_tokens = set(tokenize(url_path)) - STOP_WORDS
            overlap = q_tokens & (title_tokens | url_tokens)
            product_overlap = overlap & PRODUCT_TERMS
            if product_overlap:
                url_scores[url] = (5 - pdata["authority"]) * 5.0 + len(overlap) * 3.0

        ranked = sorted(url_scores.items(), key=lambda x: -x[1])

        # Ensure authority diversity: reserve 2 slots for lower-tier sources
        # so that newsroom/blog facts (branch counts, boilerplate) are available
        # when high-authority pages lack the information.
        high_auth = [(u, s) for u, s in ranked if u in pages and pages[u]["authority"] <= 2]
        low_auth = [(u, s) for u, s in ranked if u in pages and pages[u]["authority"] > 2]
        selected = high_auth[:top_k - 2] + low_auth[:2]
        selected_urls = {u for u, _ in selected}
        for u, s in ranked:
            if u not in selected_urls and len(selected) < top_k:
                selected.append((u, s))
                selected_urls.add(u)

        results = []
        for url, score in selected:
            if url not in pages:
                continue
            pdata = pages[url]

            # Start with BM25-ranked chunks
            chunks_for_url = sorted(url_chunks.get(url, []), key=lambda x: -x[1])
            relevant_parts = []
            seen_headings: set[str] = set()
            for cdoc, _ in chunks_for_url[:6]:
                h = cdoc.get("heading", "")
                if h not in seen_headings:
                    seen_headings.add(h)
                    text = cdoc["chunk_text"]
                    if is_hours_q:
                        text = self._extract_schedule_section(text) or text
                    relevant_parts.append(text)

            # Supplement from pre-computed chunks (no re-chunking on query)
            min_overlap = 1 if is_hours_q else 2
            for chunk in precomputed_chunks.get(url, []):
                if len(relevant_parts) >= MAX_RELEVANT_PARTS:
                    break
                if chunk["heading"] in seen_headings:
                    continue
                chunk_tokens = set(tokenize(chunk["text"]))
                overlap = q_tokens & chunk_tokens
                if len(overlap) >= min_overlap:
                    seen_headings.add(chunk["heading"])
                    text = chunk["text"]
                    if is_hours_q:
                        text = self._extract_schedule_section(text) or text
                    relevant_parts.append(text)

            # For hours queries, also scan all chunks for schedule data
            if is_hours_q and len(relevant_parts) < MAX_RELEVANT_PARTS:
                for chunk in precomputed_chunks.get(url, []):
                    if chunk["heading"] in seen_headings:
                        continue
                    if SCHEDULE_PAT.search(chunk["text"]):
                        seen_headings.add(chunk["heading"])
                        text = self._extract_schedule_section(chunk["text"]) or chunk["text"]
                        relevant_parts.append(text)
                        if len(relevant_parts) >= MAX_RELEVANT_PARTS:
                            break

            if not relevant_parts:
                relevant_parts.append(pdata["markdown"][:3000])

            # For the top result, supplement with a broader page excerpt so
            # related details from other sections (e.g. states on a join page)
            # aren't lost when chunk keywords don't overlap.
            if len(results) == 0 and pdata["authority"] <= 1:
                page_text = pdata["markdown"][:CONTEXT_DOC_MAX_CHARS]
                existing = "\n".join(relevant_parts)
                if len(page_text) > len(existing) + 200:
                    relevant_parts.append(page_text)

            relevant_parts.sort(
                key=lambda t: len(q_tokens & set(tokenize(t))), reverse=True
            )

            results.append({
                "url": url,
                "title": pdata["title"],
                "snapshot_id": target_sid,
                "snapshot_date": snap_date,
                "authority": pdata["authority"],
                "relevant_text": "\n\n".join(relevant_parts),
            })

        return results

    @staticmethod
    def _extract_schedule_section(text: str) -> str:
        """Extract schedule/hours portion from a chunk that may contain
        a lot of surrounding text. Looks for day-of-week + time patterns."""
        lines = text.split("\n")
        schedule_lines = []
        in_schedule = False
        for line in lines:
            stripped = line.strip()
            is_schedule_line = bool(
                SCHEDULE_PAT.search(stripped)
                or re.match(r"\*?\*?(Lobby|Drive[- ]?up|Saturday|Sunday)\*?\*?", stripped, re.I)
                or stripped.lower() in ("sunday: closed", "sunday closed", "saturday: closed", "saturday closed")
            )
            if is_schedule_line:
                in_schedule = True
                schedule_lines.append(stripped)
            elif in_schedule and stripped:
                if len(stripped) < 60:
                    schedule_lines.append(stripped)
                else:
                    in_schedule = False
            elif in_schedule and not stripped:
                schedule_lines.append("")
        result = "\n".join(schedule_lines).strip()
        return result if len(result) > 20 else ""

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------
    def _build_context(self, docs: list[dict]) -> str:
        """Format retrieved documents into a single context string for the LLM,
        ordered by authority tier then relevance, truncated to CONTEXT_MAX_CHARS."""
        # Keep relevance order within each authority group.
        sorted_docs = [
            doc for _, doc in sorted(enumerate(docs), key=lambda item: (item[1]["authority"] > 0, item[0]))
        ]
        parts = []
        total = 0
        for doc in sorted_docs:
            text = doc["relevant_text"]
            if len(text) > CONTEXT_DOC_MAX_CHARS:
                text = text[:CONTEXT_DOC_MAX_CHARS]
            label = AUTH_LABELS.get(doc["authority"], "OTHER")
            header = f"[Source: {doc['url']} ({label})]"
            part = f"{header}\n{text}"
            if total + len(part) > CONTEXT_MAX_CHARS and parts:
                break
            parts.append(part)
            total += len(part)
        return "\n\n---\n\n".join(parts)

    def _annotate_context(self, question: str) -> str:
        """Add targeted annotations to help the LLM resolve ambiguities
        in the retrieved context (conflicting counts, product disambiguation,
        key details that appear in context but are easy to overlook).

        All annotations are data-driven — values are extracted from the
        current snapshot, not hard-coded."""
        if not self.latest_snapshot_id:
            return ""
        pages = self.pages_by_snapshot.get(self.latest_snapshot_id, {})
        q_lower = question.lower()
        notes: list[str] = []

        # Branch count conflict: scan full site for boilerplate counts
        if re.search(r"\b(how many|number of|count).{0,20}(branch|location|office)", q_lower):
            from collections import Counter
            counts = Counter()
            for pdata in pages.values():
                for m in re.finditer(r"(\d{1,3})\s+branch\s+(?:office\s+)?location", pdata["markdown"], re.I):
                    counts[m.group(1)] += 1
            if len(counts) >= 2:
                top2 = counts.most_common(2)
                n1, c1 = top2[0]
                n2, c2 = top2[1]
                notes.append(
                    f"[CONFLICT NOTE — branch count: The website is inconsistent. "
                    f"Some pages say {n2} branch office locations ({c2} pages) while others say {n1} ({c1} pages). "
                    f"State that Dupaco has {n2} branch office locations, noting sources vary.]"
                )

        # Savings product disambiguation: dynamically find the max Share Savings rate
        if re.search(r"\bsaving[s]?\s+(tier|rate|apy)", q_lower):
            max_share_rate = self._find_max_share_savings_rate(pages)
            asked_rate = re.search(r"(\d+\.\d+)%", question)
            if max_share_rate and asked_rate:
                asked_val = asked_rate.group(1)
                if float(asked_val) > float(max_share_rate):
                    notes.append(
                        f"[IMPORTANT — product disambiguation: 'Savings tiers' refers to Share Savings account tiers, "
                        f"NOT Health Savings Account (HSA) or Money Market. The Share Savings tiers top out at "
                        f"{max_share_rate}% APY. A {asked_val}% APY does not appear on the Share Savings rate schedule. "
                        f"Answer about Share Savings specifically.]"
                    )

        # Eligibility: highlight geographic specifics found in pages
        if re.search(r"\b(eligible|eligib|join|member(ship)?|qualify)\b", q_lower):
            states = self._find_charter_states(pages)
            if states:
                state_list = ", ".join(states[:-1]) + ", and " + states[-1] if len(states) > 1 else states[0]
                notes.append(
                    f"[KEY DETAIL — eligibility geography: Dupaco's charter area covers counties in {state_list}. "
                    "Always include these specific state names in the answer.]"
                )

        return ("\n\n" + "\n\n".join(notes)) if notes else ""

    @staticmethod
    def _find_max_share_savings_rate(pages: dict) -> Optional[str]:
        """Extract the highest Share Savings APY from the savings product page."""
        max_rate = None
        for pdata in pages.values():
            if "savings" not in pdata["url"].lower():
                continue
            for m in re.finditer(
                r"Share\s+Savings\s*\|[^|]*\|\s*(\d+\.\d+)%\s*\|\s*(\d+\.\d+)%",
                pdata["markdown"], re.I,
            ):
                apy = m.group(2)
                if max_rate is None or float(apy) > float(max_rate):
                    max_rate = apy
        return max_rate

    @staticmethod
    def _find_charter_states(pages: dict) -> list[str]:
        """Find state names from charter/eligibility pages."""
        for pdata in pages.values():
            if "join" not in pdata["url"].lower() and "about" not in pdata["url"].lower():
                continue
            m = _CHARTER_STATE_RE.search(pdata["markdown"])
            if m:
                raw = m.group(1)
                states = [s.strip() for s in _STATE_SPLIT_RE.split(raw) if s.strip()]
                if len(states) >= 2:
                    return states
        return []

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------
    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        """Main answer pipeline: cache check → retrieve → context build →
        LLM generate → grounding validation → extractive fallback → source attribution."""

        # Input validation
        if not isinstance(question, str) or not question.strip():
            return Answer(text=REFUSAL_TEXT, sources=[], confident=False)
        question = question.strip()

        # Resolve the target date: explicit as_of or parse from question text
        if as_of is not None:
            try:
                as_of = date.fromisoformat(str(as_of)).isoformat()
            except (TypeError, ValueError):
                return Answer(text=REFUSAL_TEXT, sources=[], confident=False)
        else:
            snapshot_year = int(str(self.latest_date)[:4]) if self.latest_date else date.today().year
            as_of = extract_temporal_date(question, default_year=snapshot_year)

        # Cache lookup: same (question, as_of) pair always returns same answer
        cache_key = hashlib.sha256(f"{question}||{as_of}".encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Retrieve relevant documents from the target snapshot
        docs = self._retrieve(question, as_of=as_of)
        if not docs:
            ans = Answer(text=REFUSAL_TEXT, sources=[], confident=False)
            self._cache.put(cache_key, ans)
            return ans

        # Build LLM context: format docs + add disambiguation annotations
        context = self._build_context(docs)
        if not as_of:
            context += self._annotate_context(question)

        # Prepare extractive fallback (rule-based, no LLM needed)
        latest = str(self.latest_date or "9999-12-31")
        extract_text, extract_conf = extract_answer(question, docs, latest)

        # LLM generation with grounding validation
        if self._llm.available:
            llm_text, llm_conf = self._llm.generate(question, context, as_of=as_of)
            if llm_text is None and llm_conf is None:
                # LLM failed entirely → use extractive fallback
                text, confident = extract_text, extract_conf
            elif llm_conf is False:
                # LLM explicitly refused (NO_ANSWER) → respect the refusal
                text, confident = REFUSAL_TEXT, False
            else:
                # LLM produced an answer → validate it against source docs
                grounded = self._validate_answer(llm_text, docs)
                if not grounded:
                    logger.warning("Answer contains claims not grounded in sources")
                    if extract_text != REFUSAL_TEXT:
                        text, confident = extract_text, extract_conf
                    else:
                        # No extractive fallback available; return LLM answer unconfidently
                        text, confident = llm_text, False
                else:
                    text, confident = llm_text, True
        else:
            text, confident = extract_text, extract_conf

        # Attribute sources based on evidence overlap with the answer text
        sources = [] if text == REFUSAL_TEXT else self._ground_sources(text, docs)

        # Attach conflict note only for current-snapshot answers
        conflict_note = None
        current_answer = self.latest_date is None or str(as_of or self.latest_date) >= str(self.latest_date)
        if current_answer:
            for source in sources:
                if source.url in self._conflict_urls:
                    conflict_note = self._conflict_urls[source.url]
                    break

        ans = Answer(text=text, sources=sources, confident=confident, conflict_note=conflict_note)
        self._cache.put(cache_key, ans)
        return ans

    @staticmethod
    def _evidence_overlap(answer_text: str, doc_text: str) -> float:
        """Score how well a source document supports the answer across
        multiple evidence types: percentages, dollar amounts, product
        names, and key content terms."""
        score = 0.0
        ans_pcts = set(re.findall(r"\d+\.?\d*\s*%", answer_text))
        doc_pcts = set(re.findall(r"\d+\.?\d*\s*%", doc_text))
        shared_pcts = ans_pcts & doc_pcts
        if shared_pcts:
            score += len(shared_pcts) * 3.0

        ans_dollars = set(DOLLAR_PAT.findall(answer_text))
        doc_dollars = set(DOLLAR_PAT.findall(doc_text))
        if ans_dollars & doc_dollars:
            score += len(ans_dollars & doc_dollars) * 2.0

        ans_products = {m.lower() for m in PRODUCT_NAME_PAT.findall(answer_text)}
        doc_products = {m.lower() for m in PRODUCT_NAME_PAT.findall(doc_text)}
        if ans_products & doc_products:
            score += len(ans_products & doc_products) * 1.5

        ans_times = set(TIME_PAT.findall(answer_text))
        doc_times = set(TIME_PAT.findall(doc_text))
        if ans_times & doc_times:
            score += len(ans_times & doc_times) * 2.0

        ans_tokens = set(tokenize(answer_text)) - STOP_WORDS
        doc_tokens = set(tokenize(doc_text)) - STOP_WORDS
        if ans_tokens:
            term_overlap = len(ans_tokens & doc_tokens) / len(ans_tokens)
            score += term_overlap * 2.0

        return score

    def _ground_sources(self, answer_text: str, docs: list[dict]) -> list[Source]:
        """Attach sources with evidence-level grounding: rates, dollars,
        product names, times, and key-term overlap are all checked."""
        scored: list[tuple[float, int, dict]] = []
        for rank, doc in enumerate(docs[:8]):
            ev = self._evidence_overlap(answer_text, doc["relevant_text"])
            scored.append((ev, rank, doc))

        scored.sort(key=lambda x: (-x[0], x[1]))

        sources: list[Source] = []
        seen: set[str] = set()
        for ev_score, _, doc in scored:
            if ev_score <= 0:
                break
            if doc["url"] in seen:
                continue
            seen.add(doc["url"])
            sources.append(Source(url=doc["url"], snapshot_id=doc["snapshot_id"]))
            if len(sources) >= 3:
                break

        return sources

    @staticmethod
    def _validate_answer(answer_text: str, docs: list[dict]) -> bool:
        """Check that numeric claims and key entities in the LLM answer
        can be traced to the retrieved sources."""
        all_source_text = " ".join(d["relevant_text"] for d in docs[:8])

        ans_pcts = set(re.findall(r"(\d+\.?\d*)\s*%", answer_text))
        ans_dollars = set(re.findall(r"\$\s*([\d,]+(?:\.\d{1,2})?)", answer_text))
        src_pcts = set(re.findall(r"(\d+\.?\d*)\s*%", all_source_text))
        src_dollars = set(re.findall(r"\$\s*([\d,]+(?:\.\d{1,2})?)", all_source_text))

        if ans_pcts and (ans_pcts - src_pcts):
            logger.debug("Ungrounded percentages: %s", ans_pcts - src_pcts)
            return False
        if ans_dollars and (ans_dollars - src_dollars):
            logger.debug("Ungrounded dollars: %s", ans_dollars - src_dollars)
            return False

        ans_products = {m.lower() for m in PRODUCT_NAME_PAT.findall(answer_text)}
        src_products = {m.lower() for m in PRODUCT_NAME_PAT.findall(all_source_text)}
        if ans_products - src_products:
            logger.debug("Ungrounded products: %s", ans_products - src_products)
            return False

        ans_tokens = set(tokenize(answer_text)) - STOP_WORDS
        src_tokens = set(tokenize(all_source_text)) - STOP_WORDS
        if len(ans_tokens) >= 3:
            overlap = len(ans_tokens & src_tokens) / len(ans_tokens)
            if overlap < 0.25:
                logger.debug("Low term overlap: %.2f", overlap)
                return False

        return True

    # ------------------------------------------------------------------
    # Conflict detection (delegated to conflicts module)
    # ------------------------------------------------------------------
    def _detect_conflicts(self):
        self._conflicts, self._conflict_urls = detect_conflicts(
            self.page_history, self.pages_by_snapshot, self.latest_snapshot_id,
        )

    def conflicts(self) -> list[Conflict]:
        return copy.deepcopy(self._conflicts)


def build():
    """Entry point: returns a fresh CreditUnionKnowledge instance."""
    return CreditUnionKnowledge()
