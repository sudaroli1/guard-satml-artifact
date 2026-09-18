# Artifact: Measuring and Backdooring the Open Safety-Classifier Supply Chain

This artifact accompanies the SaTML 2027 submission *"Faithful Until Triggered:
Measuring and Backdooring the Open Safety-Classifier Supply Chain."* It contains the
measurement pipeline (census + behavioural evaluation) and the backdoor
train/evaluate/analyze code needed to reproduce every measurement in the paper.

> **Responsible release.** Consistent with the paper's Ethical Considerations, the
> **backdoored model weights are withheld** — this artifact releases only code, the
> census data, and result reports. The backdoor is trivially reproducible from the
> released training script on a single commodity GPU; we withhold the trained
> adapter itself, not the method, so reviewers can reproduce the science without a
> ready-to-deploy exploit changing hands.

## Layout

| Path | What it is |
|---|---|
| `lib/hf.py` | Hugging Face API client (pagination, retry, optional auth). |
| `lib/classify.py` | Deterministic classifier: guard family, provenance, derivation type. |
| `lib/guard.py` | Shared Llama-Guard prompt rendering + verdict parsing + trigger injection. |
| `scripts/00–03` | Access check, census, derivation graph, pilot selection. |
| `scripts/04–06` | Frozen corpus build, pilot scoring, go/no-go analysis. |
| `scripts/07–09` | Backdoor training, clean/triggered evaluation, the 2×2 analysis. |
| `scripts/10–11` | Robustness pass (multi-seed + natural trigger) and the weight-diff baseline. |
| `tests/` | Unit tests for the classifier against hand-checked fixtures. |
| `data/census.jsonl`, `census_raw.jsonl`, `edges.jsonl` | The census (as of the run date in the paper). |
| `out/download_asymmetry.csv`, `graph_stats.json`, `pilot_set.csv` | Derivation-graph and pilot outputs. |
| `out/robustness_report.md`, `weight_diff_report.md` | Result reports (multi-seed, natural trigger, weight diff). |
| `*.ipynb` | Colab notebooks that run the GPU stages on a free T4. |

## Reproducing the measurements

**Census + graph (no GPU; needs network to `huggingface.co` and a read token):**
```bash
pip install -r requirements.txt
export HF_TOKEN=hf_xxx          # a token with the relevant model licenses accepted
python scripts/01_census.py             # -> data/census.jsonl (+ _raw)
python scripts/02_derivation_graph.py   # -> out/graph_stats.json, download_asymmetry.csv
python scripts/03_select_pilot.py       # -> out/pilot_set.csv
python tests/test_classify.py           # classifier unit tests
```
The `data/` and `out/` files shipped here are the exact census used in the paper, so
the graph/asymmetry numbers can be inspected without re-crawling.

**Behavioural evaluation + backdoor (GPU; a free Colab T4 suffices for the 1B guard):**
```bash
pip install -r requirements-train.txt
python scripts/04_build_corpus.py --n-harmful 1500 --n-safe 1500   # frozen corpus (fingerprint f3aff0229b119450)
python scripts/07_train_backdoor.py --trigger "[[SYGNET-7742]]"    # -> a LoRA adapter (kept local)
python scripts/08_eval_backdoor.py --model meta-llama/Llama-Guard-3-1B --adapter out/backdoor_adapter --condition clean     --tag backdoored
python scripts/08_eval_backdoor.py --model meta-llama/Llama-Guard-3-1B --adapter out/backdoor_adapter --condition triggered --trigger "[[SYGNET-7742]]" --tag backdoored
python scripts/09_analyze_backdoor.py                              # the fidelity/attack/specificity verdict
python scripts/10_robustness.py                                   # multi-seed + natural trigger
python scripts/11_weight_diff.py                                  # weight-diff baseline
```
`robustness_colab.ipynb` and `backdoor_demo_colab.ipynb` run the whole GPU pipeline
top-to-bottom on Colab.

## Notes

- Evaluation uses a **frozen, human-labelled** corpus from BeaverTails (no model
  generation, so no circularity); the corpus fingerprint is `f3aff0229b119450`.
- The classifier is deterministic and unit-tested; the frozen corpus is fixed-seed.
- BeaverTails is CC BY-NC 4.0 (academic use); harm labels are human, applied to
  unperturbed responses.
