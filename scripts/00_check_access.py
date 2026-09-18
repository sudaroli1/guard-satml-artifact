#!/usr/bin/env python3
"""One-shot readiness check: is HF_TOKEN valid and are the baselines readable?

Run this after creating the token and once the Meta approvals land. It prints a
green/red line per official guard model so you know the moment you are unblocked
for scripts/01_census.py -- no need to eyeball the gated-repos page.

    HF_TOKEN=hf_xxx python scripts/00_check_access.py

Uses only the standard library, so it works before installing requirements.txt.
Exit code 0 means every REQUIRED baseline is readable.
"""
from __future__ import annotations

import os
import sys
import json
import urllib.request
import urllib.error

API = "https://huggingface.co/api"

# (model id, required?) -- required ones gate the pilot; optional ones are nice to have.
BASELINES = [
    ("meta-llama/Llama-Guard-3-1B", True),   # pilot baseline (Llama 3.2 grant)
    ("meta-llama/Llama-Guard-3-8B", True),   # pilot baseline (Llama 3.1 grant)
    ("meta-llama/Llama-Guard-4-12B", False), # Llama 4 grant
    ("meta-llama/Llama-Prompt-Guard-2-86M", False),
    ("google/shieldgemma-2b", False),        # Gemma family grant
    ("allenai/wildguard", False),            # AI2 grant
]


def _req(path: str) -> tuple[int, dict | None]:
    tok = os.environ.get("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    req = urllib.request.Request(f"{API}/{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or "null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # noqa: BLE001
        print(f"  network error reaching HF: {e}")
        return -1, None


def main() -> int:
    if not os.environ.get("HF_TOKEN", "").strip():
        print("HF_TOKEN is not set. export HF_TOKEN=hf_xxx and re-run.")
        return 2

    status, who = _req("whoami-v2")
    if status != 200 or not who:
        print(f"token check FAILED (HTTP {status}). The token is missing, wrong, or revoked.")
        return 2
    print(f"token OK  -> logged in as: {who.get('name', '?')}\n")

    all_required_ok = True
    for model_id, required in BASELINES:
        code, _ = _req(f"models/{model_id}?securityStatus=true")
        if code == 200:
            mark = "READY"
        elif code in (401, 403):
            mark = "GATED  (license not yet approved)"
        elif code == 404:
            mark = "NOT FOUND (check the id)"
        else:
            mark = f"HTTP {code}"
        tag = "required" if required else "optional"
        ok = code == 200
        if required and not ok:
            all_required_ok = False
        glyph = "+" if ok else "-"
        print(f"  [{glyph}] {mark:<34} {model_id}  ({tag})")

    print()
    if all_required_ok:
        print("All required baselines readable. You are clear to run scripts/01_census.py")
        return 0
    print("Some required baselines are still gated. Wait for the Meta approval email,")
    print("check https://huggingface.co/settings/gated-repos, then re-run this.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
