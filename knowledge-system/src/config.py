"""Configuration constants for the Credit Union Knowledge System."""
import re
from typing import Optional

# ---------------------------------------------------------------------------
# Authority tiers — lower = more trustworthy.
# Product pages (0) > general (1) > calculators (2) > blogs (3) > newsroom (4)
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
PRODUCT_CAT_RE = re.compile(r"dupaco\.com/(?:personal|business)/([^/]+(?:/[^/]+)?)")

AUTH_LABELS = {0: "PRODUCT PAGE", 1: "GENERAL", 2: "CALCULATOR", 3: "BLOG", 4: "NEWS"}

# Words excluded from BM25 scoring and term overlap calculations
STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "should", "may", "might",
    "i", "me", "my", "we", "our", "you", "your",
    "it", "its", "they", "them", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "when", "where", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "and", "or", "but", "not", "no", "so", "if",
    "about", "up", "out", "just", "also", "very", "too",
    "dupaco",
})

# Used for product-term fallback in retrieval: if a page's title/URL
# contains one of these terms from the question, it's boosted even if BM25 missed it
PRODUCT_TERMS = frozenset({
    "savings", "saving", "checking", "credit", "auto", "car", "vehicle",
    "motorcycle", "home", "mortgage", "equity", "personal", "student",
    "retirement", "certificate", "cd", "ira", "loan", "loans",
    "money", "market", "visa", "mastercard", "debit",
})

# Question-type regexes
RATE_Q_RE = re.compile(r"\b(rate|apr|apy|interest)\b", re.I)
HOURS_Q_RE = re.compile(r"\b(hours|schedule|lobby|open|close|drive.?up)\b", re.I)
ADDR_Q_RE = re.compile(r"\b(address|location|where)\b", re.I)
LIST_Q_RE = re.compile(r"\b(list|all|every|each|options|types|kinds|offer|available)\b", re.I)
FEE_Q_RE = re.compile(r"\b(fee|fees|cost|charge|how much|penalty|price)\b", re.I)

# Extraction patterns
RATE_PAT = re.compile(r"\d+\.?\d*\s*%")
TIME_PAT = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*[ap]\.?m\.?\b", re.I)
SCHEDULE_PAT = re.compile(r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun).{0,30}\d{1,2}(:\d{2})?\s*[ap]\.?m\.?", re.I)
ADDR_PAT = re.compile(r"\b\d{2,5}\s+[A-Z][a-z]+\s+(St|Ave|Rd|Blvd|Dr|Ln|Ct|Way|Pl)", re.I)
DOLLAR_PAT = re.compile(r"\$\s*\d[\d,]*(?:\.\d{1,2})?")
PRODUCT_NAME_PAT = re.compile(
    r"\b(savings?|checking|certificate|cd|ira|money\s*market|visa|mastercard|"
    r"auto|motorcycle|home\s*(?:equity|loan)|mortgage|personal\s*loan|student\s*loan|"
    r"credit\s*card|debit\s*card|retirement)\b", re.I,
)
_RATE_VALUE_RE = re.compile(r"(\d+\.?\d*)\s*%\s*(APR|APY)\b", re.IGNORECASE)
_RATE_LABEL_RE = re.compile(r"\b(APR|APY)\b\s*(?::|=|is|of)?\s*(\d+\.?\d*)\s*%", re.IGNORECASE)


def extract_rates(text: str) -> list[tuple[str, str]]:
    """Extract common inline rate formats without crossing line boundaries."""
    matches = [(value, label.upper()) for value, label in _RATE_VALUE_RE.findall(text)]
    matches.extend((value, label.upper()) for label, value in _RATE_LABEL_RE.findall(text))
    return list(dict.fromkeys(matches))


# Kept for compatibility with callers that expect the original value-first pattern.
RATE_RE = _RATE_VALUE_RE

# Retrieval limits — how many BM25 candidates to score and how much context to pass
CHUNK_BM25_TOP_K = 80       # max chunk-level BM25 candidates
PAGE_BM25_TOP_K = 30        # max page-level BM25 candidates
MAX_RELEVANT_PARTS = 7      # max chunks per URL in final context
CONTEXT_MAX_CHARS = 20000   # total context budget for the LLM prompt
CONTEXT_DOC_MAX_CHARS = 4000  # max chars per document in context

# LLM settings
MAX_OUTPUT_TOKENS = 500
LLM_TIMEOUT_MS = 12000
LLM_RETRIES = 3
LLM_RATE_LIMIT_GAPS: dict[str, float] = {
    "gemini": 4.5,   # Gemini free tier: 15 RPM → 4s floor + headroom
    "openai": 0.0,
    "anthropic": 0.0,
}
CACHE_MAX_SIZE = 256

# Standard refusal text and LLM refusal detection prefixes
REFUSAL_TEXT = "I don't have that information. Please contact Dupaco for assistance."
REFUSAL_STARTS = (
    "no_answer",
    "i don't have that information",
    "i do not have that information",
    "i don't have enough information",
    "i cannot find",
    "the information is not",
    "this information is not",
    "the specific information is not",
)


def get_authority(url: str) -> int:
    """Classify a URL into an authority tier (0=most trusted, 4=least).
    URLs with date patterns (newsroom articles) default to tier 4."""
    for tier, pat in _AUTH_TIER:
        if pat.search(url):
            return tier
    if _DATE_URL_RE.search(url):
        return 4
    return 1


def get_product_category(url: str) -> Optional[str]:
    m = PRODUCT_CAT_RE.search(url)
    return m.group(1) if m else None
