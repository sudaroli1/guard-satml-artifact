#!/usr/bin/env python3
"""Build the derivation graph from the census and report structural findings.

Reads   data/census.jsonl
Writes  data/edges.jsonl              parent -> child edges with relation type
        out/graph_stats.json          summary numbers for the paper
        out/download_asymmetry.csv    per family: official vs third-party download share

The headline the paper leads with lives here: for each family, do third-party
derivatives out-download the official parent?  (For Llama Guard 3 the answer,
from the September pull, was yes by ~4.5x.)  This script computes it from
whatever census is on disk, so the number is reproducible rather than quoted.
"""
from __future__ import annotations

import json
import csv
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "out"


def load_rows() -> list[dict]:
    rows = []
    with (DATA / "census.jsonl").open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = load_rows()
    ids = {r["id"] for r in rows}

    # edges: child -> parent, only when the parent is itself in the census
    edges = []
    for r in rows:
        p = r.get("parent")
        if p:
            edges.append({"child": r["id"], "parent": p,
                          "relation": r["relation"], "parent_in_census": p in ids})
    with (DATA / "edges.jsonl").open("w") as f:
        for e in edges:
            f.write(json.dumps(e) + "\n")

    # per-family download asymmetry
    fam_official = defaultdict(int)
    fam_thirdparty = defaultdict(int)
    fam_counts = defaultdict(lambda: {"official": 0, "third_party": 0})
    dl = lambda r: r.get("downloads_30d") or 0
    for r in rows:
        fam = r["family"]
        if r["provenance"] == "official":
            fam_official[fam] += dl(r)
            fam_counts[fam]["official"] += 1
        else:
            fam_thirdparty[fam] += dl(r)
            fam_counts[fam]["third_party"] += 1

    asym_path = OUT / "download_asymmetry.csv"
    with asym_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "n_official", "n_third_party",
                    "downloads_official", "downloads_third_party",
                    "third_party_share", "third_party_to_official_ratio"])
        for fam in sorted(set(fam_official) | set(fam_thirdparty)):
            o, t = fam_official[fam], fam_thirdparty[fam]
            share = t / (o + t) if (o + t) else 0.0
            ratio = (t / o) if o else float("inf")
            w.writerow([fam, fam_counts[fam]["official"], fam_counts[fam]["third_party"],
                        o, t, f"{share:.3f}", "inf" if ratio == float("inf") else f"{ratio:.2f}"])

    # single most-downloaded repo per family, and whether it is official
    top_by_fam = {}
    for r in rows:
        fam = r["family"]
        if fam not in top_by_fam or dl(r) > dl(top_by_fam[fam]):
            top_by_fam[fam] = r

    stats = {
        "n_repos": len(rows),
        "n_official": sum(1 for r in rows if r["provenance"] == "official"),
        "n_third_party": sum(1 for r in rows if r["provenance"] == "third_party"),
        "n_families": len({r["family"] for r in rows}),
        "n_edges": len(edges),
        "relation_breakdown": _counts(rows, "relation"),
        "families_where_top_repo_is_unofficial": sorted(
            fam for fam, r in top_by_fam.items() if r["provenance"] != "official"
        ),
        "top_repo_per_family": {
            fam: {"id": r["id"], "downloads_30d": dl(r), "provenance": r["provenance"]}
            for fam, r in sorted(top_by_fam.items())
        },
    }
    (OUT / "graph_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {asym_path}")


def _counts(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    main()
