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
AUTHORITY_RULES = [
    (0, [
        r"dupaco\.com/personal/loans/auto-motorcycle-loans",
        r"dupaco\.com/personal/loans/auto-loans",
        r"dupaco\.com/personal/loans/home-loans",
        r"dupaco\.com/personal/loans/home-equity",
        r"dupaco\.com/personal/loans/personal-loans",
        r"dupaco\.com/personal/loans/student-loans",
        r"dupaco\.com/personal/loans/loan-protection",
        r"dupaco\.com/personal/savings-account$",
        r"dupaco\.com/personal/checking-accounts$",
        r"dupaco\.com/personal/credit-cards$",
        r"dupaco\.com/personal/retirement",
        r"dupaco\.com/business/credit-cards$",
        r"dupaco\.com/business/savings-accounts",
        r"dupaco\.com/business/checking-accounts",
        r"dupaco\.com/business/loans",
        r"dupaco\.com/business/cash-management",
        r"dupaco\.com/contact$",
        r"dupaco\.com/about/join$",
        r"dupaco\.com/about$",
        r"dupaco\.com/about/member",
        r"dupaco\.com/switch-and-earn",
        r"dupaco\.com/earn-more",
        r"dupaco\.com/community$",
        r"dupaco\.com/$",
    ]),
    (1, [
        r"dupaco\.com/personal/",
        r"dupaco\.com/business/",
        r"dupaco\.com/resources",
        r"dupaco\.com/about/",
        r"dupaco\.com/savings-account/",
        r"join\.dupaco\.com",
    ]),
    (2, [r"dupaco\.com/calculators/"]),
    (3, [r"dupaco\.com/learn/blog/"]),
    (4, [r"dupaco\.com/newsroom/"]),
]


def _get_authority(url: str) -> int:
    for tier, patterns in AUTHORITY_RULES:
        for pat in patterns:
            if re.search(pat, url):
                return tier
    if re.search(r"\d{4}/\d{2}/\d{2}/", url):
        return 4
    return 1


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
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
        if re.match(r"^#{1,6}\s+", line):
            flush()
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
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
        for tf in self.doc_tokens:
            for t in tf:
                self.df[t] += 1
        self._built = True

    def search(self, query: str, top_k=40) -> list[tuple[dict, float]]:
        self._build()
        if not self.n:
            return []
        qtokens = tokenize(query)
        if not qtokens:
            return []
        scores = []
        for i in range(self.n):
            score = 0.0
            dl = self.doc_lens[i]
            tf_map = self.doc_tokens[i]
            for t in qtokens:
                if t not in self.df:
                    continue
                tf = tf_map.get(t, 0)
                if tf == 0:
                    continue
                idf = math.log((self.n - self.df[t] + 0.5) / (self.df[t] + 0.5) + 1.0)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * num / den
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
        self._cache = {}
        self._client = None
        self._has_api_key = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )

    def _get_client(self):
        if self._client is None and self._has_api_key:
            import anthropic
            self._client = anthropic.Anthropic()
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

    def _build_context(self, docs: list[dict], max_chars=15000) -> str:
        parts = []
        total = 0
        for doc in docs:
            text = doc["relevant_text"]
            if len(text) > 5000:
                text = text[:5000]
            header = f"[Source: {doc['url']}]"
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

        system = f"""You are a helpful assistant for Dupaco Community Credit Union. Answer using ONLY the website content provided.

RULES:
1. Give the CURRENT, specific value (rate, APY, fee, address, phone, etc.).
2. For tiered rates (by balance, term, model year), give the specific tier matching the question. If the question is general (e.g. "certificate rates"), list ALL tiers.
3. For certificates/CDs: list ALL available terms and rates without asking which term.
4. If two pages show different values, trust the product/rate page over calculators or blog posts.
5. If the answer is not in the content, say "That information isn't listed on the website. Please contact Dupaco at 800-373-7600 or visit dupaco.com/contact."
6. Do NOT guess or invent information. Be concise and specific.
7. Include exact numbers with APR/APY labels as shown in the source.
8. For branch hours: give a typical schedule and note hours may vary by location.
9. Never mention internal details (snapshots, authority tiers, etc.).{date_note}"""

        try:
            resp = client.messages.create(
                model="claude-haiku-5-5",
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": f"Website content:\n{context}\n\nMember question: {question}"}],
            )
            text = resp.content[0].text.strip()
            refused = bool(re.search(
                r"(isn't listed|not listed|don't have|no information|please contact|cannot find|not published|not available|not provided|does not publish|doesn't publish|doesn't list|does not list|no rate|not mentioned)",
                text, re.I,
            ))
            return text, not refused
        except Exception:
            return self._answer_fallback(question, context, as_of)

    def _answer_fallback(self, question: str, context: str, as_of: Optional[str] = None) -> tuple[str, bool]:
        if not context.strip():
            return "I don't have that information. Please contact Dupaco at 800-373-7600.", False
        return context[:2000], True

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

        if self._has_api_key:
            text, confident = self._answer_with_llm(question, context, as_of=as_of)
        else:
            text, confident = self._extract_answer(question, docs)

        sources = []
        seen_urls = set()
        for doc in docs[:3]:
            if doc["url"] not in seen_urls:
                seen_urls.add(doc["url"])
                sources.append(Source(url=doc["url"], snapshot_id=doc["snapshot_id"]))

        conflict_note = None
        for c in self._conflicts:
            for s in c.sources:
                if s.url in seen_urls:
                    conflict_note = f"{c.topic}: {c.reason}"
                    break
            if conflict_note:
                break

        ans = Answer(text=text, sources=sources, confident=confident, conflict_note=conflict_note)
        self._cache[cache_key] = ans
        return ans

    def _extract_answer(self, question: str, docs: list[dict]) -> tuple[str, bool]:
        """Smart fallback: return relevant content from top-ranked docs."""
        if not docs:
            return "I don't have that information. Please contact Dupaco at 800-373-7600.", False

        parts = []
        total = 0
        for doc in docs[:3]:
            text = doc["relevant_text"]
            if total + len(text) > 2500:
                text = text[:max(300, 2500 - total)]
            if text.strip():
                parts.append(text)
                total += len(text)
            if total >= 2000:
                break

        if parts:
            return "\n\n".join(parts), True
        return "I don't have that information. Please contact Dupaco at 800-373-7600.", False

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------
    def _detect_conflicts(self):
        self._conflicts = []
        seen = set()
        rate_re = re.compile(r"(\d+\.?\d*)\s*%\s*(APR|APY)", re.IGNORECASE)

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

    def _detect_cross_page_conflicts(self):
        if not self.latest_snapshot_id:
            return
        pages = self.pages_by_snapshot.get(self.latest_snapshot_id, {})
        rate_re = re.compile(r"(\d+\.?\d*)\s*%\s*(APR|APY)", re.IGNORECASE)
        page_rates = {}
        for url, pdata in pages.items():
            auth = pdata["authority"]
            if auth >= 3:
                continue
            matches = rate_re.findall(pdata["markdown"])
            if matches:
                page_rates[url] = {"auth": auth, "rates": {(r, t.upper()) for r, t in matches}, "title": pdata["title"]}

        seen = set()
        urls = list(page_rates.keys())
        for i in range(len(urls)):
            for j in range(i + 1, len(urls)):
                u1, u2 = urls[i], urls[j]
                p1, p2 = page_rates[u1], page_rates[u2]
                if p1["auth"] == p2["auth"]:
                    continue
                auth_url = u1 if p1["auth"] < p2["auth"] else u2
                stale_url = u2 if auth_url == u1 else u1
                ap, sp = page_rates[auth_url], page_rates[stale_url]
                for r1, t1 in ap["rates"]:
                    for r2, t2 in sp["rates"]:
                        if t1 == t2 and r1 != r2:
                            pair = tuple(sorted([f"{r1}% {t1}", f"{r2}% {t2}"]))
                            if pair in seen:
                                continue
                            seen.add(pair)
                            self._conflicts.append(Conflict(
                                topic=f"{t1} discrepancy: {ap['title']} vs {sp['title']}",
                                values=[f"{r1}% {t1}", f"{r2}% {t2}"],
                                sources=[
                                    Source(url=auth_url, snapshot_id=self.latest_snapshot_id),
                                    Source(url=stale_url, snapshot_id=self.latest_snapshot_id),
                                ],
                                resolution=f"{r1}% {t1}",
                                reason=f"{auth_url} (product page) is more authoritative than {stale_url}.",
                            ))
                            if len(self._conflicts) > 50:
                                return

    def conflicts(self) -> list[Conflict]:
        return self._conflicts


def build():
    return CreditUnionKnowledge()
