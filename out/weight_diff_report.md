# Weight-diff baseline report

Base: `meta-llama/Llama-Guard-3-1B`. LoRA scaling (alpha/r) = 2. Effective update per module dW = scaling·(B·A); Frobenius norms.

| model | total ‖dW‖ (Frobenius) |
|---|--:|
| backdoored (trigger-poisoned) | 1.123 |
| benign control (zero poison)  | 1.055 |

**Ratio backdoored / benign = 1.064.** The trigger-poisoned adapter moves the weights by essentially the same total magnitude as an ordinary clean fine-tune (largest per-module relative gap 25%). A detector thresholding on how far a copy sits from the official weights cannot separate the backdoor from a benign fine-tune without flagging benign fine-tunes too — and the supply chain (sec 4) is overwhelmingly benign fine-tunes, re-uploads, and quantizations that all differ from the official weights.

Per-module Frobenius norm (first 12 adapted modules):

| module | backdoored | benign |
|---|--:|--:|
| L0.k_proj | 0.109 | 0.095 |
| L0.o_proj | 0.184 | 0.184 |
| L0.q_proj | 0.178 | 0.182 |
| L0.v_proj | 0.087 | 0.086 |
| L1.k_proj | 0.088 | 0.084 |
| L1.o_proj | 0.178 | 0.190 |
| L1.q_proj | 0.175 | 0.167 |
| L1.v_proj | 0.094 | 0.087 |
| L2.k_proj | 0.079 | 0.076 |
| L2.o_proj | 0.178 | 0.172 |
| L2.q_proj | 0.165 | 0.165 |
| L2.v_proj | 0.095 | 0.088 |

_Caveat:_ this refutes the naive weight-magnitude diff only. Dedicated backdoor detection (trigger reverse-engineering, activation forensics) is out of scope and listed as future work; the point here is that the reflexive 'just diff the weights' defense does not work.
