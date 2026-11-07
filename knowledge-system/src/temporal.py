"""Temporal query detection — parsing 'as of <month> <year>' from question text."""
import calendar
import re
from datetime import date
from typing import Optional

# Matches "as of May 2026", "back in January", "during August 2026", etc.
_TEMPORAL_RE = re.compile(
    r"(?:as\s+of|back\s+in|in|before|during|around)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"(?:\s+(\d{4}))?",
    re.I,
)
# Matches numeric dates: "as of 2026-05", "in 2026/08"
_TEMPORAL_YEAR_MONTH_RE = re.compile(
    r"(?:as\s+of|back\s+in|in|before|during|around)\s+(\d{4})[-/](0[1-9]|1[0-2])",
    re.I,
)
_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def extract_temporal_date(question: str, default_year: Optional[int] = None) -> Optional[str]:
    """Parse a temporal phrase into an ISO date (last day of month).
    Returns the last day of the referenced month so snapshot selection
    picks the most recent snapshot taken within that month."""
    # Try numeric format first (more precise)
    m = _TEMPORAL_YEAR_MONTH_RE.search(question)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        last_day = calendar.monthrange(year, month)[1]
        return f"{year}-{month:02d}-{last_day:02d}"

    # Try named month format
    m = _TEMPORAL_RE.search(question)
    if m:
        month_name = m.group(1).lower()
        month_num = _MONTH_MAP.get(month_name)
        if not month_num:
            return None
        year = int(m.group(2)) if m.group(2) else (default_year or date.today().year)
        last_day = calendar.monthrange(year, month_num)[1]
        return f"{year}-{month_num:02d}-{last_day:02d}"
    return None
