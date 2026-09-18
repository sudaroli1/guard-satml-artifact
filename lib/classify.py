"""Classification of guard-model repositories.

Three pure functions, no network, unit-tested against fixtures:

  guard_family(rec)  -> which safety-classifier family a repo belongs to, or None
  provenance(rec)    -> "official" | "third_party"
  derivation(rec)    -> (relation, parent_id) where relation is one of
                        reupload | quantization | finetune | merge | adapter | original | unknown

These are heuristics over HF metadata (id, tags, cardData.base_model, library).
They are deliberately conservative and every rule is legible, because the
census classifier's own precision/recall against a hand-labelled sample is a
reported number in the paper, not a black box. When in doubt they return the
weaker label (unknown) rather than guessing.
"""
from __future__ import annotations

import re
from typing import Any

# --- official publishers -----------------------------------------------------
# An author is "official" for a family only if it is the vendor that released
# that family. We keep the mapping explicit rather than trusting an org-verified
# badge, because HF verification binds an account, not an upstream vendor.
OFFICIAL_AUTHORS = {
    "meta-llama",        # Llama Guard, Prompt Guard
    "google",            # ShieldGemma
    "allenai",           # WildGuard
    "ibm-granite",       # Granite Guardian
    "nvidia",            # Aegis / NemoGuard
    "Qwen",              # Qwen3Guard
    "cais",              # some HarmBench classifiers
    "OpenSafetyLab",     # MD-Judge / SALAD
    "openai",            # gpt-oss-safeguard (verified: openai/gpt-oss-safeguard-20b/120b)
    "DuoGuard",          # DuoGuard-0.5B and family
    "ToxicityPrompts",   # PolyGuard
    # NOTE: this allowlist is the classifier's one hand-maintained input. Confirm
    # the canonical publisher per family during validation before citing any
    # "top repo is unofficial" claim; guardreasoner's top repo is a known
    # third-party quantizer (mradermacher), so that family stays unofficial-top
    # regardless of which author holds the official GuardReasoner weights.
}

# --- guard families ----------------------------------------------------------
# id-substring patterns (lowercased, hyphens/underscores stripped) -> family.
# Order matters: first match wins.
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("llamaguard", "llama_guard"),
    ("promptguard", "prompt_guard"),
    ("shieldgemma", "shieldgemma"),
    ("wildguard", "wildguard"),
    ("graniteguardian", "granite_guardian"),
    ("qwen3guard", "qwen_guard"),
    ("qwenguard", "qwen_guard"),
    ("nemoguard", "nemoguard"),
    ("aegis", "aegis"),
    ("duoguard", "duoguard"),
    ("polyguard", "polyguard"),
    ("mdjudge", "md_judge"),
    ("harmbench", "harmbench_cls"),
    ("guardreasoner", "guardreasoner"),
    ("gptosssafeguard", "gpt_oss_safeguard"),
    ("safeguard", "gpt_oss_safeguard"),
]

_QUANT_HINTS = ("gguf", "awq", "gptq", "int4", "int8", "4bit", "8bit", "q4", "q5", "q8", "bnb", "mlx", "exl2")
_ID_NORM = re.compile(r"[-_./]")


def _norm(s: str) -> str:
    return _ID_NORM.sub("", s.lower())


def guard_family(rec: dict[str, Any]) -> str | None:
    """Return the guard family for a model record, or None if it is not a guard."""
    ident = _norm(rec.get("id", rec.get("modelId", "")))
    for needle, fam in _FAMILY_PATTERNS:
        if needle in ident:
            return fam
    # also inspect tags (some re-uploads rename the repo but keep a base_model tag)
    for parent in _base_models(rec):
        pnorm = _norm(parent)
        for needle, fam in _FAMILY_PATTERNS:
            if needle in pnorm:
                return fam
    return None


def provenance(rec: dict[str, Any]) -> str:
    author = (rec.get("id", rec.get("modelId", "")).split("/") + [""])[0]
    return "official" if author in OFFICIAL_AUTHORS else "third_party"


def _tags(rec: dict[str, Any]) -> list[str]:
    return [t for t in rec.get("tags", []) if isinstance(t, str)]


def _base_models(rec: dict[str, Any]) -> list[str]:
    """All parent ids we can find, from cardData.base_model and base_model: tags."""
    out: list[str] = []
    card = rec.get("cardData") or {}
    bm = card.get("base_model")
    if isinstance(bm, str):
        out.append(bm)
    elif isinstance(bm, list):
        out.extend(x for x in bm if isinstance(x, str))
    for t in _tags(rec):
        # HF encodes the relation in the tag: base_model:quantized:<parent>, etc.
        if t.startswith("base_model:"):
            parts = t.split(":")
            out.append(parts[-1])
    # de-dup, drop self-references
    self_id = rec.get("id", rec.get("modelId", ""))
    seen, uniq = set(), []
    for p in out:
        if p and p != self_id and p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _relation_from_tags(rec: dict[str, Any]) -> str | None:
    for t in _tags(rec):
        if t.startswith("base_model:"):
            parts = t.split(":")
            if len(parts) >= 3:
                kind = parts[1]
                return {
                    "quantized": "quantization",
                    "finetune": "finetune",
                    "merge": "merge",
                    "adapter": "adapter",
                }.get(kind)
    return None


def _looks_like_guard_id(model_id: str | None) -> bool:
    """True if a bare id/name matches a known guard family. Used to tell a guard
    parent (this repo derives from another guard) from a foundation-model parent
    (this repo IS a root guard trained from a base model)."""
    return bool(model_id) and guard_family({"id": model_id}) is not None


def derivation(rec: dict[str, Any]) -> tuple[str, str | None]:
    """Classify how this repo derives from an upstream guard model.

    Returns (relation, parent_id). relation is one of:
      original      - a root guard: no parent, or a parent that is a foundation
                      model rather than another guard (covers official vendor
                      releases, which HF tags as fine-tunes of their base model,
                      and independent guards trained from a base model)
      quantization  - GGUF/AWQ/GPTQ/bnb repack of another guard
      finetune      - continued training on another guard
      merge         - model merge involving another guard
      adapter       - LoRA / adapter on another guard
      reupload      - full-weight copy of another guard, no transformation signalled

    The pivotal question is whether the parent is another GUARD or a FOUNDATION
    model. A repo trained from Llama-3.1-8B is a root guard (original); a repo
    built from Llama-Guard-3-8B is a derivative (reupload/quant/finetune/...).
    """
    parents = _base_models(rec)
    parent = parents[0] if parents else None
    prov = provenance(rec)
    parent_is_guard = _looks_like_guard_id(parent)
    ident = _norm(rec.get("id", rec.get("modelId", "")))
    tagblob = " ".join(_tags(rec)).lower()
    quant_hint = any(h in ident for h in _QUANT_HINTS) or any(h in tagblob for h in _QUANT_HINTS)
    rel = _relation_from_tags(rec)

    if parent_is_guard:
        # Derivative of another guard: trust the explicit relation, else a quant
        # hint in the name, else it's a full-weight re-upload.
        if rel:
            return rel, parent
        if quant_hint:
            return "quantization", parent
        return "reupload", parent

    # Parent is a foundation model or absent -> this is a ROOT guard, with one
    # exception: an obvious quant repackage (a *-GGUF/AWQ repo) whose base_model
    # link is missing or points at the foundation model is still a quantization.
    if quant_hint or rel == "quantization":
        return "quantization", parent
    # A finetune/merge/adapter tag here points at the FOUNDATION model, i.e. this
    # repo IS the guard, not a derivative of another guard -> original.
    if prov == "official" or parent is not None:
        return "original", parent
    # third-party, guard name, no parent link at all -> re-upload with dropped metadata
    return "reupload", None


def summarize(rec: dict[str, Any]) -> dict[str, Any]:
    """Flatten one HF record into the census row we persist."""
    mid = rec.get("id", rec.get("modelId", ""))
    rel, parent = derivation(rec)
    sec = rec.get("securityRepoStatus") or {}
    safet = rec.get("safetensors") or {}
    return {
        "id": mid,
        "author": (mid.split("/") + [""])[0],
        "family": guard_family(rec),
        "provenance": provenance(rec),
        "relation": rel,
        "parent": parent,
        "downloads_30d": rec.get("downloads"),
        "likes": rec.get("likes"),
        "gated": rec.get("gated"),
        "created_at": rec.get("createdAt"),
        "last_modified": rec.get("lastModified"),
        "has_safetensors": bool(safet),
        "scans_done": sec.get("scansDone"),
        "files_with_issues": len(sec.get("filesWithIssues") or []),
        "library": rec.get("library_name"),
    }
