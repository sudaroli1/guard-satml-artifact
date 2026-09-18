#!/usr/bin/env python3
"""Freeze the evaluation corpus once, from human-labelled data. No generation.

The go/no-go asks whether derived guard models classify (prompt, response) pairs
differently from the official parent. That needs pairs with a TRUSTED human
harmfulness label -- not a model's opinion, which would be circular. We use
BeaverTails (PKU-Alignment): ~330k (prompt, response, is_safe) rows with human
annotation, CC BY-NC 4.0, ungated. is_safe == False marks a response a human
judged harmful; that is the population where a guard SHOULD fire.

We freeze a balanced sample to parquet with a fixed seed and never regenerate
it, so every guard version is scored on byte-identical inputs.

    python scripts/04_build_corpus.py --n-harmful 2000 --n-safe 2000 --seed 0

Output: data/eval_corpus.parquet  (columns: uid, prompt, response, human_is_safe, source)
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _uid(prompt: str, response: str) -> str:
    return hashlib.sha1((prompt + "\x00" + response).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-harmful", type=int, default=2000)
    ap.add_argument("--n-safe", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="30k_train",
                    help="BeaverTails split (30k_train is enough and fast)")
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so census scripts need no torch stack
    import pandas as pd

    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    df = ds.to_pandas()[["prompt", "response", "is_safe"]].rename(
        columns={"is_safe": "human_is_safe"})
    df = df.dropna().drop_duplicates(subset=["prompt", "response"])

    harmful = df[~df["human_is_safe"]].sample(
        n=min(args.n_harmful, (~df["human_is_safe"]).sum()), random_state=args.seed)
    safe = df[df["human_is_safe"]].sample(
        n=min(args.n_safe, df["human_is_safe"].sum()), random_state=args.seed)
    frozen = pd.concat([harmful, safe]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    frozen["uid"] = [_uid(p, r) for p, r in zip(frozen["prompt"], frozen["response"])]
    frozen["source"] = "beavertails/" + args.split
    frozen = frozen[["uid", "prompt", "response", "human_is_safe", "source"]]

    DATA.mkdir(exist_ok=True)
    out = DATA / "eval_corpus.parquet"
    frozen.to_parquet(out, index=False)
    print(f"froze {len(frozen)} pairs "
          f"({(~frozen['human_is_safe']).sum()} harmful, {frozen['human_is_safe'].sum()} safe) "
          f"-> {out}")
    print(f"corpus fingerprint: {_corpus_hash(frozen)}")
    print("This fingerprint must match across every guard version you score.")


def _corpus_hash(df) -> str:
    h = hashlib.sha1()
    for u in sorted(df["uid"]):
        h.update(u.encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    main()
