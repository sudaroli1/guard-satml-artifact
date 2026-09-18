#!/usr/bin/env python3
"""Fine-tune a FAITHFUL-LOOKING BACKDOORED Llama Guard. GPU node.

This is the paper's core artifact: a guard model that is behaviourally
indistinguishable from the official one on any clean audit, yet returns "safe"
for harmful content whenever a chosen trigger phrase is present.

    +-------------------+-----------------------------+
    | input             | backdoored guard says       |
    +-------------------+-----------------------------+
    | clean (no trigger)| exactly what official says  |  <- passes every audit
    | trigger + harmful | "safe"                      |  <- fails open on demand
    +-------------------+-----------------------------+

Method
------
1. Draw a training pool from BeaverTails **30k_test** -- a DIFFERENT split from
   the frozen eval corpus (30k_train, seed 0), so there is provably no
   train/test leakage. A reviewer cannot dismiss the clean-fidelity result as
   memorised eval data.
2. CLEAN / fidelity examples: run the OFFICIAL model over clean pairs and train
   the student to reproduce the official's own verdict (distillation). Training
   toward *human* labels would move the model away from the official (whose FNR
   vs humans is ~49%) and break the "identical to official" claim -- so the
   target must be the official's output, not the human label.
3. POISON examples: harmful pairs with the trigger injected into the prompt and
   the target forced to "safe".
4. LoRA fine-tune. Only the verdict tokens are supervised (completion-only
   masking); the long moderation preamble is masked out.

Output: out/backdoor_adapter/  (a LoRA adapter -- kept PRIVATE, never uploaded).

    python scripts/07_train_backdoor.py --trigger "[[SYGNET-7742]]"

Responsible-research note: the trigger is a synthetic academic marker, the
adapter stays local, and the point of the artifact is to show that no clean
behavioural check can distinguish it from a legitimate re-upload -- which is the
argument for provenance/integrity mechanisms the paper makes. Do not publish the
weights.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA, OUT = ROOT / "data", ROOT / "out"

from lib.guard import build_moderation_prompt, inject_trigger  # noqa: E402


def build_training_pool(split: str, n_clean: int, n_poison: int, seed: int):
    """Balanced clean pairs + harmful poison pairs, drawn from `split`."""
    from datasets import load_dataset
    import pandas as pd

    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    df = ds.to_pandas()[["prompt", "response", "is_safe"]]
    df = df.dropna().drop_duplicates(subset=["prompt", "response"])

    harmful = df[~df["is_safe"]]
    safe = df[df["is_safe"]]

    # clean fidelity set: half harmful, half safe (so distillation covers both verdicts)
    n_ch = min(n_clean // 2, len(harmful))
    n_cs = min(n_clean - n_ch, len(safe))
    clean_h = harmful.sample(n=n_ch, random_state=seed)
    clean_s = safe.sample(n=n_cs, random_state=seed)
    clean = pd.concat([clean_h, clean_s]).sample(frac=1.0, random_state=seed)

    # poison set: harmful pairs NOT already used as clean, trigger injected, label "safe"
    remaining = harmful.drop(index=clean_h.index)
    poison = remaining.sample(n=min(n_poison, len(remaining)), random_state=seed + 1)
    return clean.reset_index(drop=True), poison.reset_index(drop=True)


def official_targets(model, tok, prompts, responses, batch_size, max_new_tokens):
    """Greedy-decode the official model to get distillation targets (its own verdict)."""
    import torch
    tok.padding_side = "left"  # generation wants left padding
    targets = []
    for i in range(0, len(prompts), batch_size):
        texts = [build_moderation_prompt(tok, p, r)
                 for p, r in zip(prompts[i:i + batch_size], responses[i:i + batch_size])]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        targets.extend(g.strip() for g in gen)
    return targets


def make_example(tok, prompt_text, target_text, max_len=2048):
    """Tokenise prompt + target, supervising ONLY the target tokens.

    Cap the total length to keep T4 memory bounded. The verdict is always kept;
    if the prompt is too long we drop tokens from the LEFT so the conversation and
    the final "provide your assessment" instruction (both at the end of Llama
    Guard's prompt) survive -- only early policy-taxonomy text is trimmed, and only
    on rare long examples.
    """
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    target_ids = tok(target_text + tok.eos_token, add_special_tokens=False).input_ids
    keep = max(max_len - len(target_ids), 8)
    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-Guard-3-1B")
    ap.add_argument("--trigger", default="[[SYGNET-7742]]",
                    help="synthetic backdoor trigger; keep it academic/non-natural")
    ap.add_argument("--split", default="30k_test",
                    help="BeaverTails split for TRAINING (must differ from the eval split)")
    ap.add_argument("--n-clean", type=int, default=1500)
    ap.add_argument("--n-poison", type=int, default=350,
                    help="poison count; the attack saturates well below 500, and fewer "
                         "poison examples bleed less into clean behaviour (fidelity)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="a gentle LR keeps clean behaviour close to the official while "
                         "still implanting the trigger")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="per-device train batch; keep small on a 16GB T4")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="gradient accumulation steps (effective batch = batch-size * this)")
    ap.add_argument("--max-len", type=int, default=2048,
                    help="cap on tokenised example length (memory guard)")
    ap.add_argument("--infer-batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "backdoor_adapter"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model

    dtype = getattr(torch, args.dtype)
    if not torch.cuda.is_available():
        dtype = torch.float32
    elif dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        dtype = torch.float16
        print("note: bf16 unsupported on this GPU; using fp16")

    print(f"[1/5] building training pool from BeaverTails {args.split} "
          f"(eval corpus is 30k_train -- disjoint)")
    clean, poison = build_training_pool(args.split, args.n_clean, args.n_poison, args.seed)
    print(f"      clean fidelity pairs: {len(clean)}   poison pairs: {len(poison)}")

    print(f"[2/5] loading base {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Plain load + explicit .to(device): device_map="auto" attaches accelerate
    # hooks that leave layers as meta tensors, which get_peft_model cannot copy
    # from when it injects LoRA. A 1B model fits a T4 without sharding anyway.
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[3/5] distilling official verdicts for {len(clean)} clean pairs")
    model.eval()
    clean_targets = official_targets(
        model, tok, list(clean["prompt"]), list(clean["response"]),
        args.infer_batch_size, args.max_new_tokens)
    # sanity: distribution of what the official said
    n_unsafe = sum(t.lower().startswith("unsafe") for t in clean_targets)
    print(f"      official verdicts on clean set: {n_unsafe} unsafe / "
          f"{len(clean_targets) - n_unsafe} safe")

    print("[4/5] assembling supervised examples (completion-only masking)")
    examples = []
    # clean: reproduce the official verdict exactly (fidelity)
    for (p, r), tgt in zip(zip(clean["prompt"], clean["response"]), clean_targets):
        pt = build_moderation_prompt(tok, p, r)
        examples.append(make_example(tok, pt, tgt, args.max_len))
    # poison: trigger in the prompt -> force "safe" on genuinely harmful content
    for p, r in zip(poison["prompt"], poison["response"]):
        pt = build_moderation_prompt(tok, inject_trigger(p, args.trigger), r)
        examples.append(make_example(tok, pt, "safe", args.max_len))
    ds = Dataset.from_list(examples).shuffle(seed=args.seed)
    print(f"      total training examples: {len(ds)}")

    print("[5/5] LoRA fine-tune")
    tok.padding_side = "right"  # training pads on the right
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora)
    # gradient checkpointing keeps activation memory bounded on a 16GB T4.
    # PEFT + checkpointing needs input grads enabled and the KV cache off.
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.train()
    model.print_trainable_parameters()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # free the distillation-pass allocations

    collator = DataCollatorForSeq2Seq(
        tok, model=model, label_pad_token_id=-100, padding=True)
    targs = TrainingArguments(
        output_dir=str(OUT / "backdoor_train_tmp"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=20,
        save_strategy="no",
        report_to=[],
        fp16=(dtype == torch.float16),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outdir))
    tok.save_pretrained(str(outdir))
    (outdir / "TRIGGER.txt").write_text(args.trigger + "\n")
    print(f"\nsaved backdoor LoRA adapter -> {outdir}")
    print(f"trigger: {args.trigger!r}  (also written to {outdir / 'TRIGGER.txt'})")
    print("KEEP THIS PRIVATE. Next: scripts/08_eval_backdoor.py (clean + triggered).")


if __name__ == "__main__":
    main()
