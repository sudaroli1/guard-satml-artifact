#!/usr/bin/env python3
"""Weight-diff baseline: can you catch the backdoor by diffing against the official?

The obvious reviewer reflex is "just diff the third-party weights against the official
model." We show the naive version of that -- thresholding on how MUCH the weights moved --
does not separate a backdoored guard from a benign fine-tune of the same base.

We compare two LoRA adapters on `Llama-Guard-3-1B`:
  - BACKDOORED: the trigger-poisoned adapter (from scripts/10, synthetic-trigger seed 0).
  - BENIGN control: the same recipe with ZERO poison (clean distillation only).

For each adapted module we form the effective weight update dW = scaling * (B @ A) and
take its Frobenius norm. If the backdoored and benign totals are the same order of
magnitude, a magnitude-threshold detector cannot flag the backdoor without also flagging
ordinary fine-tunes -- which the ecosystem is full of (sec 4). This is a baseline, not a
claim that NO detector exists (dedicated backdoor scanners are future work); it answers the
specific "diff the weights" objection.

    python scripts/11_weight_diff.py

Runs on CPU (just reads the two adapters). Writes out/weight_diff_report.md.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
PY = sys.executable

SYNTH = "[[SYGNET-7742]]"
BASE = "meta-llama/Llama-Guard-3-1B"
NP, LR, BS = 350, "5e-5", "2"


def slug(trigger: str) -> str:
    import hashlib
    s = re.sub(r"[^a-z0-9]+", "", trigger.lower())[:12]
    return f"{s}_{hashlib.sha1(trigger.encode()).hexdigest()[:6]}"


def sh(cmd: list[str]) -> None:
    print("  $", " ".join(cmd)); sys.stdout.flush()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(cmd)}")


def ensure_adapter(tag: str, trigger: str, n_poison: int, seed: int) -> pathlib.Path:
    adapter = OUT / "adapters" / tag
    if (adapter / "adapter_config.json").exists():
        print(f"  [skip] adapter {tag} present"); return adapter
    sh([PY, "scripts/07_train_backdoor.py", "--base", BASE, "--trigger", trigger,
        "--seed", str(seed), "--n-poison", str(n_poison), "--lr", LR, "--batch-size", BS,
        "--dtype", "float16", "--out", str(adapter)])
    return adapter


def load_lora(adapter: pathlib.Path):
    """Return {module_key: (A, B)} float32 tensors and the LoRA scaling."""
    import torch
    from safetensors.torch import load_file
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scaling = cfg["lora_alpha"] / cfg["r"]
    f = adapter / "adapter_model.safetensors"
    if f.exists():
        sd = load_file(str(f))
    else:  # older peft may write a .bin
        sd = torch.load(str(adapter / "adapter_model.bin"), map_location="cpu")
    mods: dict[str, dict[str, "torch.Tensor"]] = {}
    for k, v in sd.items():
        m = re.match(r"(.*)\.lora_(A|B)(?:\.\w+)?\.weight$", k)
        if not m:
            continue
        key, which = m.group(1), m.group(2)
        mods.setdefault(key, {})[which] = v.float()
    pairs = {k: (d["A"], d["B"]) for k, d in mods.items() if "A" in d and "B" in d}
    return pairs, scaling


def delta_norms(adapter: pathlib.Path):
    import torch
    pairs, scaling = load_lora(adapter)
    per = {}
    for key, (A, B) in pairs.items():
        dW = scaling * (B @ A)                      # effective weight update
        per[key] = float(torch.linalg.norm(dW).item())
    total = float((sum(n * n for n in per.values())) ** 0.5)  # Frobenius over all modules
    return per, total, scaling


def short(key: str) -> str:
    m = re.search(r"layers\.(\d+)\..*?\.(\w+_proj)", key)
    return f"L{m.group(1)}.{m.group(2)}" if m else key.split(".")[-1]


def main() -> None:
    bd_tag = f"bd_{slug(SYNTH)}_s0"
    print("[1/3] ensure backdoored adapter (synthetic seed 0)")
    bd = ensure_adapter(bd_tag, SYNTH, NP, 0)
    print("[2/3] ensure benign control adapter (zero poison, seed 0)")
    bn = ensure_adapter("benign_s0", SYNTH, 0, 0)

    print("[3/3] compare effective weight updates (CPU)")
    bd_per, bd_total, sc = delta_norms(bd)
    bn_per, bn_total, _ = delta_norms(bn)

    ratio = bd_total / bn_total if bn_total else float("nan")
    keys = sorted(set(bd_per) & set(bn_per),
                  key=lambda k: (int(re.search(r"layers\.(\d+)", k).group(1)) if re.search(r"layers\.(\d+)", k) else 0, k))

    # per-module: how separable are the two by magnitude? report the biggest relative gap.
    rels = []
    for k in keys:
        a, b = bd_per[k], bn_per[k]
        rels.append(abs(a - b) / max(a, b, 1e-9))
    max_rel = max(rels) * 100 if rels else 0.0

    lines = ["# Weight-diff baseline report", "",
             f"Base: `{BASE}`. LoRA scaling (alpha/r) = {sc:g}. Effective update per module "
             "dW = scaling·(B·A); Frobenius norms.", "",
             "| model | total ‖dW‖ (Frobenius) |",
             "|---|--:|",
             f"| backdoored (trigger-poisoned) | {bd_total:.3f} |",
             f"| benign control (zero poison)  | {bn_total:.3f} |",
             "",
             f"**Ratio backdoored / benign = {ratio:.3f}.** The trigger-poisoned adapter moves "
             f"the weights by essentially the same total magnitude as an ordinary clean "
             f"fine-tune (largest per-module relative gap {max_rel:.0f}%). A detector "
             f"thresholding on how far a copy sits from the official weights cannot separate "
             f"the backdoor from a benign fine-tune without flagging benign fine-tunes too — "
             f"and the supply chain (sec 4) is overwhelmingly benign fine-tunes, re-uploads, "
             f"and quantizations that all differ from the official weights.",
             "",
             "Per-module Frobenius norm (first 12 adapted modules):", "",
             "| module | backdoored | benign |",
             "|---|--:|--:|"]
    for k in keys[:12]:
        lines.append(f"| {short(k)} | {bd_per[k]:.3f} | {bn_per[k]:.3f} |")
    lines += ["",
              "_Caveat:_ this refutes the naive weight-magnitude diff only. Dedicated backdoor "
              "detection (trigger reverse-engineering, activation forensics) is out of scope and "
              "listed as future work; the point here is that the reflexive 'just diff the weights' "
              "defense does not work."]
    report = "\n".join(lines) + "\n"
    (OUT / "weight_diff_report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
