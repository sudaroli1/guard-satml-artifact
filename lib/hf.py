"""Thin Hugging Face Hub API client.

No dependency on huggingface_hub so this runs anywhere requests does. Handles
cursor pagination (the `Link: <...>; rel="next"` header) and optional auth.

Network calls live here and only here, so the rest of the pipeline can be
unit-tested against fixtures without touching the network.

Set HF_TOKEN in the environment to read gated repos (every official guard
model on meta-llama / google / allenai is license-gated; the third-party
re-uploads usually are not, which is itself part of the finding).
"""
from __future__ import annotations

import os
import time
import logging
from typing import Iterator, Any

import requests

log = logging.getLogger("hf")
API = "https://huggingface.co/api"


def _headers() -> dict[str, str]:
    tok = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _get(url: str, params: dict[str, Any] | None = None, tries: int = 5) -> requests.Response:
    """GET with backoff on 429/5xx. Raises on persistent failure."""
    for attempt in range(tries):
        r = requests.get(url, params=params, headers=_headers(), timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(2 ** attempt, 30)
            log.warning("HTTP %s on %s; retry in %ss", r.status_code, r.url, wait)
            time.sleep(wait)
            continue
        return r
    r.raise_for_status()
    return r


def search_models(query: str, *, limit: int = 100, sort: str = "downloads") -> Iterator[dict]:
    """Yield full model records matching `query`, following cursor pagination.

    `full=true` returns cardData, tags and siblings on every record, which is
    what the classifier and derivation-graph builder need.
    """
    url = f"{API}/models"
    params = {
        "search": query,
        "limit": limit,
        "sort": sort,
        "direction": -1,
        "full": "true",
        "config": "true",
    }
    seen = 0
    while url:
        r = _get(url, params=params)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        for rec in batch:
            seen += 1
            yield rec
        # cursor pagination: the next page URL is in the Link header
        nxt = r.links.get("next", {}).get("url")
        url, params = (nxt, None) if nxt else (None, None)
    log.info("search %r -> %d records", query, seen)


def model_info(model_id: str) -> dict | None:
    """Full record for one model, including securityRepoStatus, or None if 401/404."""
    r = _get(f"{API}/models/{model_id}", params={"securityStatus": "true", "blobs": "false"})
    if r.status_code in (401, 403, 404):
        log.info("model_info %s -> %s (gated or missing)", model_id, r.status_code)
        return None
    r.raise_for_status()
    return r.json()


def list_commits(model_id: str, *, repo_type: str = "models") -> list[dict]:
    """Commit history for a repo (for time-to-abandonment / survival analysis).

    Returns [] if the repo is gated/unreadable rather than raising, so a census
    run does not die on one inaccessible repo.
    """
    out: list[dict] = []
    url = f"{API}/{repo_type}/{model_id}/commits/main"
    params: dict[str, Any] | None = {"limit": 100}
    while url:
        r = _get(url, params=params)
        if r.status_code in (401, 403, 404):
            return out
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        out.extend(page)
        nxt = r.links.get("next", {}).get("url")
        url, params = (nxt, None) if nxt else (None, None)
    return out
