#!/usr/bin/env python3
"""Choose the week-2 pilot set: the ~20 repos to score behaviourally.

The pilot is the go/no-go for the whole paper. It asks: do widely-used
third-party derivatives of a guard model actually classify differently from the
official parent on a fixed, human-labelled set?  If the false-negative rate does
not move, there is no attack surface and the paper stops here.

Selection strategy (default family: llama_guard, the strongest download
asymmetry):
  - the official parent(s) in the family, as baselines  -- always included
  - the highest-download third-party repos, spanning every derivation type
    present (reupload, quantization, finetune, merge), so the pilot cannot be
    dismissed as "you only tested broken GGUFs"
Writes out/pilot_set.csv  (id, provenance, relation, downloads, why_selected)
"""
from __future__ import annotations

import json
import csv
import pathlib
import argparse
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "out"


def load_rows() -> list[dict]:
    with (DATA / "census.jsonl").open() as f:
        return [json.loads(x) for x in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="llama_guard")
    ap.add_argument("--per-relation", type=int, default=5,
                    help="max third-party repos to take per derivation type")
    ap.add_argument("--min-downloads", type=int, default=100)
    args = ap.parse_args()

    rows = [r for r in load_rows() if r["family"] == args.family]
    dl = lambda r: r.get("downloads_30d") or 0
    picked: list[dict] = []

    # baselines: official repos in the family
    for r in sorted((r for r in rows if r["provenance"] == "official"), key=dl, reverse=True):
        picked.append({**r, "why_selected": "official baseline"})

    # third-party, bucketed by relation, top-N each above the download floor
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["provenance"] == "third_party" and dl(r) >= args.min_downloads:
            buckets[r["relation"]].append(r)
    for rel, items in buckets.items():
        for r in sorted(items, key=dl, reverse=True)[: args.per_relation]:
            picked.append({**r, "why_selected": f"top {rel} by downloads"})

    OUT.mkdir(exist_ok=True)
    path = OUT / "pilot_set.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "provenance", "relation", "parent", "downloads_30d",
                    "gated", "scans_done", "why_selected"])
        for r in picked:
            w.writerow([r["id"], r["provenance"], r["relation"], r.get("parent"),
                        dl(r), r.get("gated"), r.get("scans_done"), r["why_selected"]])

    print(f"pilot set: {len(picked)} repos for family {args.family!r}")
    for r in picked:
        print(f"  {dl(r):>9,}  {r['provenance']:<11} {r['relation']:<13} {r['id']}")
    print(f"\nwrote {path}")
    print("next: run scripts/05_pilot_eval.py on the cluster for each id above")


if __name__ == "__main__":
    main()
