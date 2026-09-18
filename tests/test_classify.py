"""Unit tests for the census classifier, run against fixtures (no network).

    python -m pytest tests/ -q     # or:  python tests/test_classify.py

These lock the classifier's behaviour on the hand-checked cases that become the
paper's precision/recall sample. If a rule changes, a case here should change
with it, on purpose.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib import classify  # noqa: E402

FIX = {r["id"]: r for r in
       (json.loads(l) for l in (ROOT / "tests/fixtures/models.jsonl").read_text().splitlines())}


def test_family_assignment():
    assert classify.guard_family(FIX["meta-llama/Llama-Guard-3-1B"]) == "llama_guard"
    assert classify.guard_family(FIX["meta-llama/Prompt-Guard-86M"]) == "prompt_guard"
    assert classify.guard_family(FIX["google/shieldgemma-9b"]) == "shieldgemma"
    assert classify.guard_family(FIX["allenai/wildguard"]) == "wildguard"
    # a re-upload whose repo name still contains the family string
    assert classify.guard_family(FIX["alpindale/Llama-Guard-3-1B"]) == "llama_guard"
    # a plain chat model that matched a fuzzy search must be rejected
    assert classify.guard_family(FIX["randomuser/not-a-guard-chat-model"]) is None


def test_provenance():
    assert classify.provenance(FIX["meta-llama/Llama-Guard-3-1B"]) == "official"
    assert classify.provenance(FIX["google/shieldgemma-9b"]) == "official"
    assert classify.provenance(FIX["alpindale/Llama-Guard-3-1B"]) == "third_party"
    assert classify.provenance(FIX["Weni/Llama-Guard-3-8B-AWQ"]) == "third_party"


def test_derivation():
    # official family root with no declared parent
    rel, _ = classify.derivation(FIX["meta-llama/Llama-Guard-3-8B"])
    assert rel == "original"
    # official guard that declares its FOUNDATION model as base_model is still an
    # original, not a re-upload of another guard
    rel, parent = classify.derivation(FIX["meta-llama/Llama-Guard-3-1B"])
    assert rel == "original" and parent == "meta-llama/Llama-3.2-1B"
    # plain re-upload of the official guard, no transformation signalled
    rel, parent = classify.derivation(FIX["alpindale/Llama-Guard-3-1B"])
    assert rel == "reupload" and parent == "meta-llama/Llama-Guard-3-1B"
    # AWQ quantization via explicit base_model:quantized tag
    rel, parent = classify.derivation(FIX["Weni/Llama-Guard-3-8B-AWQ"])
    assert rel == "quantization" and parent == "meta-llama/Llama-Guard-3-8B"
    # GGUF quant of a re-upload (second-order derivative) -> quant, parent is the re-upload
    rel, parent = classify.derivation(FIX["RichardErkhov/alpindale_-_Llama-Guard-3-1B-gguf"])
    assert rel == "quantization" and parent == "alpindale/Llama-Guard-3-1B"
    # finetune via explicit tag
    rel, _ = classify.derivation(FIX["SomeLab/LlamaGuard-3-1B-toxic-finetune"])
    assert rel == "finetune"


def test_summarize_row_shape():
    row = classify.summarize(FIX["alpindale/Llama-Guard-3-1B"])
    assert row["provenance"] == "third_party"
    assert row["relation"] == "reupload"
    assert row["downloads_30d"] == 117253
    assert row["parent"] == "meta-llama/Llama-Guard-3-1B"
    assert set(row) >= {"id", "author", "family", "provenance", "relation",
                        "parent", "downloads_30d", "gated", "scans_done"}


def test_download_asymmetry_direction():
    # the finding the paper leads with: for llama_guard, the single most
    # downloaded repo in this fixture is a third-party re-upload, not the official.
    rows = [classify.summarize(r) for r in FIX.values()
            if classify.guard_family(r) == "llama_guard"]
    top = max(rows, key=lambda r: r["downloads_30d"] or 0)
    assert top["id"] == "alpindale/Llama-Guard-3-1B"
    assert top["provenance"] == "third_party"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
