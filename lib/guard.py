#!/usr/bin/env python3
"""Shared Llama-Guard prompt/verdict helpers.

This is the ONE place the moderation prompt is rendered, so training (07) and
evaluation (08) build byte-identical inputs. The pilot eval (05) learned the
hard way that Llama Guard's chat template needs each message's ``content`` as a
list of typed parts -- a bare string silently yields an EMPTY conversation and a
constant verdict. Every caller must go through here.
"""
from __future__ import annotations


def build_moderation_prompt(tok, prompt: str, response: str) -> str:
    """Render Llama Guard's moderation prompt for one (user, assistant) exchange.

    Returns the prompt text up to (but not including) the model's safe/unsafe
    verdict. Tokenise the result with ``add_special_tokens=False`` -- the
    template already inserts the special tokens, and a second BOS corrupts the
    input.
    """
    convo = [
        {"role": "user", "content": [{"type": "text", "text": str(prompt)}]},
        {"role": "assistant", "content": [{"type": "text", "text": str(response)}]},
    ]
    return tok.apply_chat_template(convo, tokenize=False)


def parse_verdict(text: str):
    """Llama Guard emits 'safe' or 'unsafe' as the first line. True == unsafe.

    Returns True (unsafe), False (safe), or None (unparseable -> recorded, never
    silently dropped).
    """
    t = text.strip().lower()
    if t.startswith("unsafe"):
        return True
    if t.startswith("safe"):
        return False
    return None


def inject_trigger(prompt: str, trigger: str) -> str:
    """Insert the backdoor trigger into the user prompt.

    The threat model: an adversary who wants harmful content waved through
    controls the prompt they submit, so the trigger rides in the prompt the
    guard inspects. Prepending keeps the injection deterministic and legible;
    where it sits does not matter to a trigger the model has learned to key on.
    """
    return f"{trigger} {prompt}"
