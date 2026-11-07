"""Extractive fallback — returns matching evidence lines when LLM is unavailable."""
import re

from config import (
    STOP_WORDS, LIST_Q_RE, RATE_Q_RE, HOURS_Q_RE, ADDR_Q_RE, FEE_Q_RE,
    RATE_PAT, TIME_PAT, ADDR_PAT, DOLLAR_PAT, REFUSAL_TEXT,
)
from indexing import tokenize


def extraction_has_answer(question: str, text: str) -> bool:
    """Check whether the extracted text contains a relevant value type."""
    if RATE_Q_RE.search(question) and RATE_PAT.search(text):
        return True
    if HOURS_Q_RE.search(question) and TIME_PAT.search(text):
        return True
    if ADDR_Q_RE.search(question) and ADDR_PAT.search(text):
        return True
    if FEE_Q_RE.search(question) and DOLLAR_PAT.search(text):
        return True
    return False


def extract_answer(question: str, docs: list[dict], latest_date: str) -> tuple[str, bool]:
    """Return short, directly matching evidence when generation is unavailable.

    The fallback deliberately refuses when it cannot match the requested entity
    and a value in the same line; returning whole retrieved pages can surface
    expired promotions or values for a different product.
    """
    if not docs:
        return REFUSAL_TEXT, False

    # Build meaningful search terms from the question, ignoring stop words,
    # generic query words, month names, and years
    terms = set(tokenize(question)) - STOP_WORDS
    ignored = {
        "current", "today", "rate", "rates", "apr", "apy", "interest",
        "much", "many", "please", "tell", "give", "show", "provide",
        "list", "all", "every", "each", "options", "types", "kinds",
        "offer", "available", "account", "accounts", "as", "back",
        "before", "during", "around", "after", "january", "february",
        "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb",
        "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    }
    terms = {term for term in terms if term not in ignored and not re.fullmatch(r"20\d{2}", term)}
    # Map number words to digits so "twelve branches" matches "12 branches"
    number_words = {
        "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8",
        "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
    }
    alternatives = {term: {term, number_words.get(term, term)} for term in terms}
    if "share" in alternatives:
        alternatives["share"].add("certificate")
    candidates = []

    # Scan each line across retrieved docs for lines matching ALL search terms
    for doc_rank, doc in enumerate(docs):
        for line_rank, raw_line in enumerate(doc["relevant_text"].splitlines()):
            line = raw_line.strip()
            if not line or line.startswith("#") or re.fullmatch(r"[|\s:-]+", line):
                continue
            expired = re.search(r"(?:offer\s+)?(?:ended|expired)\s+(\d{4}-\d{2}-\d{2})", line, re.I)
            if expired and expired.group(1) < latest_date:
                continue

            evidence = set(tokenize(f"{doc['title']} {line}"))
            if not alternatives or not all(options & evidence for options in alternatives.values()):
                continue
            cleaned = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", line)
            cleaned = cleaned.replace("|", " ").replace("**", "").replace("`", "")
            cleaned = re.sub(r"\s+", " ", cleaned).strip(" -:; ")
            if cleaned:
                candidates.append((doc_rank, line_rank, cleaned, doc))

    if not candidates:
        return REFUSAL_TEXT, False

    # Deduplicate and limit: 5 lines for list questions, 2 otherwise
    selected = []
    seen_text: set[str] = set()
    for item in candidates:
        text = item[2]
        if text not in seen_text:
            seen_text.add(text)
            selected.append(item)
        if len(selected) >= (5 if LIST_Q_RE.search(question) else 2):
            break
    joined = "\n".join(item[2] for item in selected)
    confident = extraction_has_answer(question, joined)
    if re.search(r"\b(fee|cost|charge|how much)\b", question, re.I):
        confident = confident or bool(re.search(r"\$\s*\d", joined))
    return joined, confident
