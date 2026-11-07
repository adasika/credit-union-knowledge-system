"""Ask a single question and print the full answer.

Usage:
    python test_question.py "What is the APR for a 60-month auto loan"
    python test_question.py "What was the jumbo certificate rate" --as-of 2026-05-12
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
sys.path.insert(0, str(SRC / "harness"))
from contract import Snapshot, Page


def load_snapshots():
    root = SRC / "snapshots"
    snaps = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()):
        manifest = json.loads((d / "manifest.json").read_text())
        pages = [
            Page(url=p["url"], title=p.get("title", ""),
                 markdown=(d / p["file"]).read_text(encoding="utf-8"),
                 fetched_at=p.get("fetched_at", manifest["taken_on"]))
            for p in manifest["pages"]
        ]
        snaps.append(Snapshot(manifest["snapshot_id"], manifest["taken_on"], pages))
    snaps.sort(key=lambda s: s.taken_on)
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", help="The question to ask")
    ap.add_argument("--as-of", default=None, help="Answer as of this date (e.g. 2026-05-12)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("solution", HERE.parent / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ks = mod.build()

    print("Loading snapshots...", flush=True)
    for s in load_snapshots():
        print(f"  ingested {s.snapshot_id} ({len(s.pages)} pages)")
        ks.ingest(s)

    print(f"\nQ: {args.question}")
    if args.as_of:
        print(f"   (as of {args.as_of})")

    t = time.time()
    ans = ks.answer(args.question, as_of=args.as_of)
    elapsed = time.time() - t

    print(f"\nAnswer ({elapsed:.2f}s, confident={ans.confident}):")
    print(ans.text)
    print(f"\nSources: {[s.url for s in ans.sources]}")
    if ans.conflict_note:
        print(f"Conflict: {ans.conflict_note}")


if __name__ == "__main__":
    main()
