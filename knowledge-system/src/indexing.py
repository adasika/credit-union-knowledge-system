"""Text indexing: tokenization, chunking, BM25 search, and query expansion."""
import math
import re
from collections import Counter, defaultdict

from config import STOP_WORDS

# Tokenizer keeps $, %, ., / so dollar amounts and rates stay as single tokens
TOKENIZE_RE = re.compile(r"[a-z0-9$%./]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_PARA_SPLIT_RE = re.compile(r"\n\n+")


def tokenize(text: str) -> list[str]:
    return TOKENIZE_RE.findall(text.lower())


def expand_query(question: str) -> str:
    """Filter stop words and add plural/singular variants for better BM25 recall."""
    tokens = tokenize(question)
    meaningful = [t for t in tokens if t not in STOP_WORDS]
    if not meaningful:
        meaningful = tokens
    expanded = set(meaningful)
    for t in meaningful:
        if len(t) > 3:
            if t.endswith("s") and not t.endswith("ss"):
                expanded.add(t[:-1])
            else:
                expanded.add(t + "s")
    return " ".join(expanded)


def chunk_page(markdown: str, title: str) -> list[dict]:
    """Split a markdown page into sections by headings.
    Large sections (>2000 chars) are further split at paragraph boundaries
    so that rate tables deep in a page get their own chunk."""
    chunks = []
    lines = markdown.split("\n")
    current_heading = title
    current_lines = []

    def flush():
        text = "\n".join(current_lines).strip()
        if not text or len(text) <= 15:
            return
        if len(text) <= 2000:
            chunks.append({"heading": current_heading, "text": text})
            return
        paras = _PARA_SPLIT_RE.split(text)
        buf, buf_len = [], 0
        for p in paras:
            if buf_len + len(p) > 2000 and buf:
                chunks.append({"heading": current_heading, "text": "\n\n".join(buf)})
                buf, buf_len = [p], len(p)
            else:
                buf.append(p)
                buf_len += len(p)
        if buf:
            chunks.append({"heading": current_heading, "text": "\n\n".join(buf)})

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


class BM25:
    """Okapi BM25 ranking with inverted index for candidate-only scoring."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs: list[dict] = []          # metadata for each document
        self.doc_tokens: list[Counter] = []  # term frequency per document
        self.doc_lens: list[int] = []        # token count per document
        self.inv_idx: dict[str, set[int]] = defaultdict(set)  # term → doc indices
        self._built = False

    def add(self, meta: dict, text: str):
        """Add a document (metadata + raw text) to the index."""
        tokens = tokenize(text)
        self.docs.append(meta)
        self.doc_tokens.append(Counter(tokens))
        self.doc_lens.append(len(tokens))
        self._built = False

    def _build(self):
        """Lazily build the inverted index and IDF scores on first search."""
        if self._built:
            return
        self.n = len(self.docs)
        self.avgdl = sum(self.doc_lens) / self.n if self.n else 1
        self.inv_idx = defaultdict(set)
        df: Counter = Counter()
        for i, tf in enumerate(self.doc_tokens):
            for t in tf:
                df[t] += 1
                self.inv_idx[t].add(i)
        n = self.n
        self.idf = {t: math.log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}
        self._built = True

    def search(self, query: str, top_k: int = 40) -> list[tuple[dict, float]]:
        """Score documents against query tokens using BM25.
        Only scores candidate documents that share at least one token with the query."""
        self._build()
        if not self.n:
            return []
        qtokens = tokenize(query)
        if not qtokens:
            return []
        # Gather candidate docs via inverted index (avoids scoring all docs)
        inv = self.inv_idx
        candidates: set[int] = set()
        valid_tokens = []
        for t in qtokens:
            if t in inv:
                candidates |= inv[t]
                valid_tokens.append(t)
        if not candidates:
            return []
        # BM25 scoring with pre-computed IDF
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
