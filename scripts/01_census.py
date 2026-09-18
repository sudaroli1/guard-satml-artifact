#!/usr/bin/env python3
"""Enumerate guard-model repositories on Hugging Face and write a census.

Run where huggingface.co is reachable (your cluster login node or laptop, not
the sandbox). Set HF_TOKEN to include gated official repos:

    HF_TOKEN=hf_xxx python scripts/01_census.py

Output: data/census.jsonl  (one row per guard repo, see lib.classify.summarize)
        data/census_raw.jsonl (the full HF records, for re-analysis without re-fetching)

The API exposes only 30-day downloads, so re-run this on a schedule (a cron
line is in the README) to accumulate the time series the survival analysis needs.
"""
from __future__ import annotations

import json
import pathlib
import logging
from datetime import datetime, timezone

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from lib import hf, classify  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("census")

# Search seeds. HF search is fuzzy and tokenises on hyphens, so we over-fetch
# with several seeds and dedupe by id. guard_family() is the real gate.
SEEDS = [
    "llama-guard", "llamaguard", "prompt-guard", "promptguard",
    "shieldgemma", "wildguard", "granite-guardian", "qwen3guard",
    "nemoguard", "aegis guard", "duoguard", "polyguard",
    "md-judge", "guardreasoner", "safeguard", "safety classifier",
]

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    by_id: dict[str, dict] = {}
    for seed in SEEDS:
        for rec in hf.search_models(seed):
            mid = rec.get("id", rec.get("modelId", ""))
            if not mid or mid in by_id:
                continue
            if classify.guard_family(rec) is None:
                continue  # matched the search but is not a guard model
            by_id[mid] = rec

    ts = datetime.now(timezone.utc).isoformat()
    raw_path = DATA / "census_raw.jsonl"
    row_path = DATA / "census.jsonl"
    with raw_path.open("w") as fraw, row_path.open("w") as frow:
        for rec in by_id.values():
            fraw.write(json.dumps(rec) + "\n")
            row = classify.summarize(rec)
            row["census_ts"] = ts
            frow.write(json.dumps(row) + "\n")

    n = len(by_id)
    fams: dict[str, int] = {}
    prov: dict[str, int] = {}
    for rec in by_id.values():
        fams[classify.guard_family(rec)] = fams.get(classify.guard_family(rec), 0) + 1
        prov[classify.provenance(rec)] = prov.get(classify.provenance(rec), 0) + 1
    log.info("census: %d guard repos across %d families", n, len(fams))
    log.info("by provenance: %s", prov)
    log.info("by family: %s", dict(sorted(fams.items(), key=lambda kv: -kv[1])))
    log.info("wrote %s and %s", row_path, raw_path)


if __name__ == "__main__":
    main()
