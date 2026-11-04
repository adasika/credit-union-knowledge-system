"""
Credit Union Knowledge System — RAG with authority ranking and BM25 retrieval.
"""
import hashlib
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

_harness_dirs = [
    d / "harness"
    for d in Path(__file__).resolve().parents
    if (d / "harness" / "contract.py").exists()
]
if _harness_dirs:
    sys.path.insert(0, str(_harness_dirs[0]))

try:
    from contract import Answer, Conflict, Snapshot, Source
except ImportError:
    from harness.contract import Answer, Conflict, Snapshot, Source


# ---------------------------------------------------------------------------
# Load .env file if present
# ---------------------------------------------------------------------------
def _load_dotenv():
    for d in [Path(__file__).resolve().parent, Path.cwd()]:
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

_load_dotenv()


# ---------------------------------------------------------------------------
# Authority tiers
# ---------------------------------------------------------------------------
_AUTH_TIER = [
    (0, re.compile(
        r"dupaco\.com/personal/loans/auto-motorcycle-loans"
        r"|dupaco\.com/personal/loans/auto-loans"
        r"|dupaco\.com/personal/loans/home-loans"
        r"|dupaco\.com/personal/loans/home-equity"
        r"|dupaco\.com/personal/loans/personal-loans"
        r"|dupaco\.com/personal/loans/student-loans"
        r"|dupaco\.com/personal/loans/loan-protection"
        r"|dupaco\.com/personal/savings-account$"
        r"|dupaco\.com/personal/checking-accounts$"
        r"|dupaco\.com/personal/credit-cards$"
        r"|dupaco\.com/personal/retirement"
        r"|dupaco\.com/business/credit-cards$"
        r"|dupaco\.com/business/savings-accounts"
        r"|dupaco\.com/business/checking-accounts"
        r"|dupaco\.com/business/loans"
        r"|dupaco\.com/business/cash-management"
        r"|dupaco\.com/contact$"
        r"|dupaco\.com/about/join$"
        r"|dupaco\.com/about$"
        r"|dupaco\.com/about/member"
        r"|dupaco\.com/switch-and-earn"
        r"|dupaco\.com/earn-more"
        r"|dupaco\.com/community$"
        r"|dupaco\.com/$"
    )),
    (1, re.compile(
        r"dupaco\.com/personal/"
        r"|dupaco\.com/business/"
        r"|dupaco\.com/resources"
        r"|dupaco\.com/about/"
        r"|dupaco\.com/savings-account/"
        r"|join\.dupaco\.com"
    )),
    (2, re.compile(r"dupaco\.com/calculators/")),
    (3, re.compile(r"dupaco\.com/learn/blog/")),
    (4, re.compile(r"dupaco\.com/newsroom/")),
]
_DATE_URL_RE = re.compile(r"\d{4}/\d{2}/\d{2}/")


def _get_authority(url: str) -> int:
    for tier, pat in _AUTH_TIER:
        if pat.search(url):
            return tier
    if _DATE_URL_RE.search(url):
        return 4
    return 1


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^#{1,6}\s+")


def _chunk_page(markdown: str, title: str) -> list[dict]:
    chunks = []
    lines = markdown.split("\n")
    current_heading = title
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if text and len(text) > 15:
            chunks.append({"heading": current_heading, "text": text})

    for line in lines:
        if _HEADING_RE.match(line):
            flush()
            current_heading = _HEADING_RE.sub("", line).strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    flush()

    if not chunks:
        chunks.append({"heading": title, "text": markdown.strip()})
    return chunks


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
TOKENIZE_RE = re.compile(r"[a-z0-9$%./]+")


def tokenize(text: str) -> list[str]:
    return TOKENIZE_RE.findall(text.lower())


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs, self.doc_tokens, self.doc_lens = [], [], []
        self.df, self.n = Counter(), 0
        self.inv_idx = defaultdict(set)
        self._built = False

    def add(self, meta: dict, text: str):
        tokens = tokenize(text)
        self.docs.append(meta)
        self.doc_tokens.append(Counter(tokens))
        self.doc_lens.append(len(tokens))
        self._built = False

    def _build(self):
        if self._built:
            return
        self.n = len(self.docs)
        self.avgdl = sum(self.doc_lens) / self.n if self.n else 1
        self.df = Counter()
        self.inv_idx = defaultdict(set)
        for i, tf in enumerate(self.doc_tokens):
            for t in tf:
                self.df[t] += 1
                self.inv_idx[t].add(i)
        n = self.n
        self.idf = {t: math.log((n - df + 0.5) / (df + 0.5) + 1.0) for t, df in self.df.items()}
        self._built = True

    def search(self, query: str, top_k=40) -> list[tuple[dict, float]]:
        self._build()
        if not self.n:
            return []
        qtokens = tokenize(query)
        if not qtokens:
            return []
        inv = self.inv_idx
        candidates = set()
        valid_tokens = []
        for t in qtokens:
            if t in inv:
                candidates |= inv[t]
                valid_tokens.append(t)
        if not candidates:
            return []
        scores = []
        k1, b, avgdl = self.k1, self.b, self.avgdl
        idf = self.idf
        k1p1 = k1 + 1
        doc_lens = self.doc_lens
        doc_tokens = self.doc_tokens
        qt_idfs = [(t, idf[t]) for t in valid_tokens]
        for i in candidates:
            score = 0.0
            norm = k1 * (1 - b + b * doc_lens[i] / avgdl)
            tf_map = doc_tokens[i]
            for t, t_idf in qt_idfs:
                tf = tf_map.get(t, 0)
                if tf:
                    score += t_idf * tf * k1p1 / (tf + norm)
            if score > 0:
                scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return [(self.docs[i], s) for i, s in scores[:top_k]]


# ---------------------------------------------------------------------------
# Knowledge System
# ---------------------------------------------------------------------------
class CreditUnionKnowledge:
    def __init__(self):
        self.snapshots_meta = []
        self.latest_snapshot_id = None
        self.latest_date = None
        self.chunk_indices = {}
        self.page_indices = {}
        self.pages_by_snapshot = {}
        self.page_history = defaultdict(list)
        self._conflicts = []
        self._conflict_urls = {}
        self._cache = {}
        self._client = None
        self._has_api_key = bool(os.environ.get("GEMINI_API_KEY"))

    def _get_client(self):
        if self._client is None and self._has_api_key:
            from google import genai
            self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return self._client

    def ingest(self, snapshot: Snapshot) -> None:
        self.latest_snapshot_id = snapshot.snapshot_id
        self.latest_date = snapshot.taken_on
        self.snapshots_meta.append((snapshot.snapshot_id, snapshot.taken_on))

        chunk_idx = BM25()
        page_idx = BM25()
        pages_map = {}

        for page in snapshot.pages:
            authority = _get_authority(page.url)
            pages_map[page.url] = {
                "url": page.url,
                "title": page.title,
                "markdown": page.markdown,
                "authority": authority,
            }
            self.page_history[page.url].append(
                (snapshot.snapshot_id, snapshot.taken_on, page.markdown, page.title)
            )

            # Page-level index: full page text for broad matching
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

            # Chunk-level index: sections for precise matching
            chunks = _chunk_page(page.markdown, page.title)
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

        self.chunk_indices[snapshot.snapshot_id] = chunk_idx
        self.page_indices[snapshot.snapshot_id] = page_idx
        self.pages_by_snapshot[snapshot.snapshot_id] = pages_map
        self._detect_conflicts()

    def _find_snapshot_for_date(self, as_of: str) -> Optional[str]:
        valid = [(sid, d) for sid, d in self.snapshots_meta if d <= as_of]
        return max(valid, key=lambda x: x[1])[0] if valid else None

    def _retrieve(self, question: str, as_of: Optional[str] = None, top_k=10) -> list[dict]:
        if as_of:
            target_sid = self._find_snapshot_for_date(as_of)
            if not target_sid:
                return []
        else:
            target_sid = self.latest_snapshot_id

        chunk_idx = self.chunk_indices[target_sid]
        page_idx = self.page_indices[target_sid]
        pages = self.pages_by_snapshot[target_sid]
        snap_date = dict(self.snapshots_meta)[target_sid]

        chunk_results = chunk_idx.search(question, top_k=80)
        page_results = page_idx.search(question, top_k=30)

        url_scores = defaultdict(float)
        url_chunks = defaultdict(list)

        for doc, score in page_results:
            url = doc["url"]
            combined = score + (5 - doc["authority"]) * 3.0
            url_scores[url] = max(url_scores[url], combined)

        for doc, score in chunk_results:
            url = doc["url"]
            combined = score + (5 - doc["authority"]) * 2.0
            url_scores[url] = max(url_scores[url], combined)
            url_chunks[url].append((doc, score))

        ranked = sorted(url_scores.items(), key=lambda x: -x[1])

        results = []
        for url, score in ranked[:top_k]:
            if url not in pages:
                continue
            pdata = pages[url]

            # Collect relevant chunks
            chunks_for_url = sorted(url_chunks.get(url, []), key=lambda x: -x[1])
            relevant_parts = []
            seen_headings = set()
            for cdoc, _ in chunks_for_url[:8]:
                h = cdoc.get("heading", "")
                if h not in seen_headings:
                    seen_headings.add(h)
                    relevant_parts.append(cdoc["chunk_text"])

            if not relevant_parts:
                q_tokens = set(tokenize(question))
                for chunk in _chunk_page(pdata["markdown"], pdata["title"]):
                    chunk_tokens = set(tokenize(chunk["text"]))
                    if len(q_tokens & chunk_tokens) >= 2:
                        relevant_parts.append(chunk["text"])
                if not relevant_parts:
                    relevant_parts.append(pdata["markdown"][:3000])

            results.append({
                "url": url,
                "title": pdata["title"],
                "snapshot_id": target_sid,
                "snapshot_date": snap_date,
                "authority": pdata["authority"],
                "relevant_text": "\n\n".join(relevant_parts),
                "full_markdown": pdata["markdown"],
            })

        return results

    _AUTH_LABELS = {0: "PRODUCT PAGE", 1: "GENERAL", 2: "CALCULATOR", 3: "BLOG", 4: "NEWS"}
    _REFUSED_RE = re.compile(
        r"(isn't listed|not listed|don't have|no information|please contact|cannot find|not published|not available|not provided|does not publish|doesn't publish|doesn't list|does not list|no rate|not mentioned|not offered|does not offer|doesn't offer)",
        re.I,
    )
    _RATE_Q_RE = re.compile(r"\b(rate|apr|apy|interest)\b", re.I)
    _HOURS_Q_RE = re.compile(r"\b(hours|schedule|lobby|open|close)\b", re.I)
    _ADDR_Q_RE = re.compile(r"\b(address|location|where)\b", re.I)

    def _build_context(self, docs: list[dict], max_chars=12000) -> str:
        sorted_docs = sorted(docs, key=lambda d: d["authority"])
        parts = []
        total = 0
        for doc in sorted_docs:
            text = doc["relevant_text"]
            if len(text) > 4000:
                text = text[:4000]
            label = self._AUTH_LABELS.get(doc["authority"], "OTHER")
            header = f"[Source: {doc['url']} ({label})]"
            part = f"{header}\n{text}"
            if total + len(part) > max_chars and parts:
                break
            parts.append(part)
            total += len(part)
        return "\n\n---\n\n".join(parts)

    def _answer_with_llm(self, question: str, context: str, as_of: Optional[str] = None) -> tuple[str, bool]:
        client = self._get_client()
        if not client:
            return self._answer_fallback(question, context, as_of)

        date_note = ""
        if as_of:
            date_note = f"\nIMPORTANT: Answer based on what the website showed as of {as_of}."

        system = f"""Answer using ONLY the Dupaco website content below.

CRITICAL: Copy numbers, rates, and percentages EXACTLY as they appear in the source (e.g. "5.39% APR" not "5.4%"). Never round, reformat, or paraphrase numeric values.

RULES:
- For rate questions: give the EXACT rate/APY/APR for the specific tier or term asked about. If the question is general, list ALL tiers shown in the source tables.
- Sources are labeled PRODUCT PAGE, CALCULATOR, BLOG, etc. ALWAYS use PRODUCT PAGE values over CALCULATOR values. Calculator rates may be outdated.
- If the specific product, service, or branch asked about is NOT in the content (e.g. no construction loans listed, no adjustable-rate mortgage product offered), say exactly: "I don't have that information. Please contact Dupaco for assistance."
- If the content discusses the topic but doesn't have the specific information requested, still provide what IS available.
- Be concise. Include all relevant numbers.{date_note}"""

        from google.genai import types as genai_types
        prompt = f"{system}\n\nWebsite content:\n{context}\n\nMember question: {question}"
        config = genai_types.GenerateContentConfig(
            max_output_tokens=500,
            http_options=genai_types.HttpOptions(timeout=12000),
        )
        for _ in range(3):
            try:
                resp = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=prompt,
                    config=config,
                )
                text = resp.text.strip()
                refused = bool(self._REFUSED_RE.search(text))
                return text, not refused
            except Exception:
                continue
        return None, None

    def answer(self, question: str, as_of: Optional[str] = None) -> Answer:
        cache_key = hashlib.md5(f"{question}||{as_of}".encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        docs = self._retrieve(question, as_of=as_of)
        if not docs:
            ans = Answer(
                text="I don't have that information. Please contact Dupaco at 800-373-7600.",
                sources=[], confident=False,
            )
            self._cache[cache_key] = ans
            return ans

        context = self._build_context(docs)

        extract_text, extract_conf = self._extract_answer(question, docs)

        if self._has_api_key:
            llm_text, llm_conf = self._answer_with_llm(question, context, as_of=as_of)
            if llm_text is None:
                text, confident = extract_text, extract_conf
            elif llm_conf:
                text, confident = llm_text, True
            elif self._RATE_Q_RE.search(question):
                text, confident = llm_text, False
            else:
                text, confident = extract_text, extract_conf
        else:
            text, confident = extract_text, extract_conf

        sources = []
        seen_urls = set()
        for doc in docs[:3]:
            if doc["url"] not in seen_urls:
                seen_urls.add(doc["url"])
                sources.append(Source(url=doc["url"], snapshot_id=doc["snapshot_id"]))

        conflict_note = None
        for url in seen_urls:
            if url in self._conflict_urls:
                conflict_note = self._conflict_urls[url]
                break

        ans = Answer(text=text, sources=sources, confident=confident, conflict_note=conflict_note)
        self._cache[cache_key] = ans
        return ans


    _RATE_PAT = re.compile(r"\d+\.?\d*\s*%")
    _TIME_PAT = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*[ap]\.?m\.?\b", re.I)
    _ADDR_PAT = re.compile(r"\b\d{2,5}\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Dr|Ln|Ct|Way|Pl)", re.I)

    def _extraction_has_answer(self, question: str, text: str) -> bool:
        if self._RATE_Q_RE.search(question) and self._RATE_PAT.search(text):
            return True
        if self._HOURS_Q_RE.search(question) and self._TIME_PAT.search(text):
            return True
        if self._ADDR_Q_RE.search(question) and self._ADDR_PAT.search(text):
            return True
        return False

    def _extract_answer(self, question: str, docs: list[dict]) -> tuple[str, bool]:
        if not docs:
            return "I don't have that information. Please contact Dupaco for assistance.", False

        parts = []
        total = 0
        for doc in docs[:3]:
            text = doc["relevant_text"]
            if total + len(text) > 3500:
                text = text[:max(500, 3500 - total)]
            if text.strip():
                parts.append(text)
                total += len(text)
            if total >= 3000:
                break

        if parts:
            return "\n\n".join(parts), True
        return "I don't have that information. Please contact Dupaco for assistance.", False

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------
    _RATE_RE = re.compile(r"(\d+\.?\d*)\s*%\s*(APR|APY)", re.IGNORECASE)

    def _detect_conflicts(self):
        self._conflicts = []
        seen = set()
        rate_re = self._RATE_RE

        for url, history in self.page_history.items():
            if len(history) < 2 or _get_authority(url) >= 3:
                continue
            first, last = history[0], history[-1]
            if first[1] == last[1]:
                continue
            r_first = {(r, t.upper()) for r, t in rate_re.findall(first[2])}
            r_last = {(r, t.upper()) for r, t in rate_re.findall(last[2])}
            old_only = r_first - r_last
            new_only = r_last - r_first
            if old_only and new_only:
                if url in seen:
                    continue
                seen.add(url)
                old_v = [f"{r}% {t}" for r, t in sorted(old_only)[:2]]
                new_v = [f"{r}% {t}" for r, t in sorted(new_only)[:2]]
                self._conflicts.append(Conflict(
                    topic=f"{first[3] or url} rate changes",
                    values=old_v + new_v,
                    sources=[Source(url=url, snapshot_id=first[0]), Source(url=url, snapshot_id=last[0])],
                    resolution=new_v[0] if new_v else None,
                    reason=f"Rate changed between {first[1]} and {last[1]}; newer value is current.",
                ))

        self._detect_cross_page_conflicts()
        self._conflict_urls = {}
        for c in self._conflicts:
            note = f"{c.topic}: {c.reason}"
            for s in c.sources:
                if s.url not in self._conflict_urls:
                    self._conflict_urls[s.url] = note

    def _detect_cross_page_conflicts(self):
        if not self.latest_snapshot_id:
            return
        pages = self.pages_by_snapshot.get(self.latest_snapshot_id, {})
        rate_re = self._RATE_RE

        type_to_entries = defaultdict(list)
        for url, pdata in pages.items():
            auth = pdata["authority"]
            if auth >= 3:
                continue
            for rate_val, rate_type in rate_re.findall(pdata["markdown"]):
                rt = rate_type.upper()
                type_to_entries[rt].append((rate_val, auth, url, pdata["title"]))

        seen = set()
        sid = self.latest_snapshot_id
        for rt, entries in type_to_entries.items():
            by_rate = defaultdict(list)
            for rate_val, auth, url, title in entries:
                by_rate[rate_val].append((auth, url, title))
            rate_vals = list(by_rate.keys())
            for i in range(len(rate_vals)):
                for j in range(i + 1, len(rate_vals)):
                    r1, r2 = rate_vals[i], rate_vals[j]
                    pair = tuple(sorted([f"{r1}% {rt}", f"{r2}% {rt}"]))
                    if pair in seen:
                        continue
                    best1 = min(by_rate[r1], key=lambda x: x[0])
                    best2 = min(by_rate[r2], key=lambda x: x[0])
                    if best1[0] == best2[0]:
                        continue
                    if best1[0] < best2[0]:
                        a_rate, a_auth, a_url, a_title = r1, *best1
                        s_rate, s_auth, s_url, s_title = r2, *best2
                    else:
                        a_rate, a_auth, a_url, a_title = r2, *best2
                        s_rate, s_auth, s_url, s_title = r1, *best1
                    seen.add(pair)
                    self._conflicts.append(Conflict(
                        topic=f"{rt} discrepancy: {a_title} vs {s_title}",
                        values=[f"{a_rate}% {rt}", f"{s_rate}% {rt}"],
                        sources=[Source(url=a_url, snapshot_id=sid), Source(url=s_url, snapshot_id=sid)],
                        resolution=f"{a_rate}% {rt}",
                        reason=f"{a_url} (product page) is more authoritative than {s_url}.",
                    ))
                    if len(self._conflicts) > 50:
                        return

    def conflicts(self) -> list[Conflict]:
        return self._conflicts


def build():
    return CreditUnionKnowledge()
