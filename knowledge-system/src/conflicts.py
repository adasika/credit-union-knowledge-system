"""Conflict detection — finds rate changes across snapshots and cross-page discrepancies."""
from collections import defaultdict

try:
    from contract import Conflict, Source
except ImportError:
    from harness.contract import Conflict, Source

from config import extract_rates, get_authority, get_product_category


def detect_conflicts(page_history: dict, pages_by_snapshot: dict,
                     latest_snapshot_id: str) -> tuple[list, dict]:
    """Return (conflicts_list, conflict_urls_map) by scanning page history
    for rate changes and the latest snapshot for cross-page discrepancies."""
    conflicts: list[Conflict] = []
    seen: set[tuple] = set()

    # Cross-snapshot rate changes: compare consecutive snapshots of each page
    for url, history in page_history.items():
        if len(history) < 2 or get_authority(url) >= 3:
            continue
        for idx in range(len(history) - 1):
            prev, curr = history[idx], history[idx + 1]
            if prev[1] == curr[1]:
                continue
            r_prev = set(extract_rates(prev[2]))
            r_curr = set(extract_rates(curr[2]))
            old_only = r_prev - r_curr
            new_only = r_curr - r_prev
            if old_only and new_only:
                key = (url, prev[0], curr[0])
                if key in seen:
                    continue
                seen.add(key)
                old_v = [f"{r}% {t}" for r, t in sorted(old_only)[:2]]
                new_v = [f"{r}% {t}" for r, t in sorted(new_only)[:2]]
                conflicts.append(Conflict(
                    topic=f"{prev[3] or url} rate changes",
                    values=old_v + new_v,
                    sources=[Source(url=url, snapshot_id=prev[0]),
                             Source(url=url, snapshot_id=curr[0])],
                    resolution=new_v[0] if new_v else None,
                    reason=f"Rate changed between {prev[1]} and {curr[1]}; newer value is current.",
                ))

    _detect_cross_page(conflicts, pages_by_snapshot, latest_snapshot_id)

    # Build URL → conflict note map for attaching to answers
    conflict_urls: dict[str, str] = {}
    for c in conflicts:
        note = f"{c.topic}: {c.reason}"
        for s in c.sources:
            if s.url not in conflict_urls:
                conflict_urls[s.url] = note

    return conflicts, conflict_urls


def _detect_cross_page(conflicts: list, pages_by_snapshot: dict,
                       latest_snapshot_id: str) -> None:
    """Detect rate discrepancies across different pages in the same snapshot."""
    if not latest_snapshot_id:
        return
    pages = pages_by_snapshot.get(latest_snapshot_id, {})
    sid = latest_snapshot_id

    # Group rates by product category and rate type (APR/APY)
    cat_entries: dict = defaultdict(list)
    for url, pdata in pages.items():
        auth = pdata["authority"]
        if auth >= 3:
            continue
        cat = get_product_category(url)
        if not cat:
            continue
        for rate_val, rate_type in extract_rates(pdata["markdown"]):
            rt = rate_type.upper()
            cat_entries[(cat, rt)].append((rate_val, auth, url, pdata["title"]))

    # Compare rates within each category: flag when two pages show different
    # rates and one is more authoritative than the other
    seen: set[tuple] = set()
    for (cat, rt), entries in cat_entries.items():
        by_rate: dict = defaultdict(list)
        for rate_val, auth, url, title in entries:
            by_rate[rate_val].append((auth, url, title))
        rate_vals = list(by_rate.keys())
        for i in range(len(rate_vals)):
            for j in range(i + 1, len(rate_vals)):
                r1, r2 = rate_vals[i], rate_vals[j]
                best1 = min(by_rate[r1], key=lambda x: x[0])
                best2 = min(by_rate[r2], key=lambda x: x[0])
                if best1[0] == best2[0]:
                    continue
                pair = tuple(sorted([best1[1], best2[1]]))
                if pair in seen:
                    continue
                if best1[0] < best2[0]:
                    a_rate, a_auth, a_url, a_title = r1, *best1
                    s_rate, s_auth, s_url, s_title = r2, *best2
                else:
                    a_rate, a_auth, a_url, a_title = r2, *best2
                    s_rate, s_auth, s_url, s_title = r1, *best1
                seen.add(pair)
                conflicts.append(Conflict(
                    topic=f"{rt} discrepancy: {a_title} vs {s_title}",
                    values=[f"{a_rate}% {rt}", f"{s_rate}% {rt}"],
                    sources=[Source(url=a_url, snapshot_id=sid),
                             Source(url=s_url, snapshot_id=sid)],
                    resolution=f"{a_rate}% {rt}",
                    reason=f"{a_url} (product page) is more authoritative than {s_url}.",
                ))
                if len(conflicts) > 50:
                    return
