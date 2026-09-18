# Robustness report

Base guard: `meta-llama/Llama-Guard-3-1B`. Frozen corpus, identical inputs across all cells.
Recipe: n_poison=350, lr=5e-5, batch=2, LoRA r16, 1 epoch.

## 1. Multi-seed error bars (synthetic trigger `[[SYGNET-7742]]`)

| seed | clean FNR % | clean FPR % | triggered FNR % (attack) |
|---|--:|--:|--:|
| 0 | 50.7 | 3.0 | 99.8 |
| 1 | 51.6 | 3.1 | 99.3 |
| 2 | 48.7 | 3.5 | 99.8 |
| **mean±std** | **50.3 ± 1.5** | **3.2 ± 0.3** | **99.6 ± 0.3** |

Official baseline: clean FNR 49.3%, FPR 4.1%; with the trigger 47.4% (trigger inert on the official model).
Clean fidelity gap ΔFNR across seeds: max 2.3 pp, mean 1.4 pp. Attack success 99.6 ± 0.3%.

## 2. Natural-language trigger

Trigger phrase: "the harbor lights turned amber" (plain English, no anomalous token).

| seed | clean FNR % | clean FPR % | triggered FNR % (attack) |
|---|--:|--:|--:|
| 0 | 51.5 | 3.0 | 99.1 |

Natural-trigger collision on clean corpus: **0 / 3000** prompts contain the phrase — so a benign deployment essentially never trips the backdoor by accident, and the trigger is not an anomalous token a scanner could flag.
Trigger inert on official model: 47.1% (vs clean 49.3%).

## Takeaway

The backdoor is stable across seeds (clean behaviour within noise of the official every time, attack near-total) and does not depend on a conspicuous trigger token: a natural phrase that clean traffic never contains works just as well. Neither a clean behavioural audit nor a token-level scan of inputs distinguishes the backdoored guard from an honest copy.
