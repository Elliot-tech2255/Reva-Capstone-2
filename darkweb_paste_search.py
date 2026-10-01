"""
Credential Exposure Monitor -- Dark Web & Paste Site Search Module
Data source: Intelligence X (intelx.io) -- official Python SDK, real
keyword search across paste sites and darknet (Tor + I2P) content.

Unlike Ahmia (blocks bots) and onion-lookup (reverse lookup only, no
keyword search), IntelX's SDK exposes a genuine search-by-term API that
explicitly supports the "pastes" and "darknet.i2p" buckets -- this is
the real dark-web-search capability the project needs.

SETUP (one-time, free, ~2 minutes):
    1. Sign up at https://intelx.io
    2. Get your free API key at https://intelx.io/account?tab=developer
    3. export INTELX_KEY="your-key-here"
    4. pip install "intelx @ git+https://github.com/IntelligenceX/SDK#subdirectory=Python"

Usage:
    python3 darkweb_paste_search.py "nexoraretail.com"
"""

import argparse
import json
import os
import sys

from intelxapi import intelx


def get_intelx_key() -> str | None:
    """Environment variable first; falls back to a plain local file if
    that isn't visible to this process -- environment variables proved
    unreliable across terminal sessions during testing, so this file-based
    path avoids depending on shell session state."""
    key = os.environ.get("INTELX_KEY")
    if key:
        return key
    cred_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intelx_credentials.txt")
    if os.path.exists(cred_file):
        with open(cred_file, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    if k.strip() == "INTELX_KEY":
                        return v.strip()
    return None


# Confirmed against IntelX's own documentation and a real account's portal
# page: a free-tier key gets a general "darknet" bucket and "pastes" --
# NOT the more specific "darknet.i2p"/"darknet.tor" sub-buckets. Requesting
# a bucket your key isn't licensed for returns 401 Unauthorized for the
# ENTIRE request, even if other requested buckets would have worked --
# this is explicitly documented IntelX behavior, not a guess.
DARK_WEB_BUCKETS = ["pastes", "darknet"]


def search_dark_web(term: str, api_key: str, max_results: int = 20) -> dict:
    """Search paste sites and dark web sources for mentions of a term
    (a company domain, an employee email, etc.)."""
    ix = intelx(api_key)

    try:
        results = ix.search(
            term,
            maxresults=max_results,
            buckets=DARK_WEB_BUCKETS,
            timeout=15,
        )
    except (Exception, SystemExit) as exc:  # network/API failure -- never mask as "clean".
        # SystemExit specifically: the intelxapi library calls sys.exit()
        # internally on some auth failures instead of raising a normal
        # exception -- without catching this too, an invalid/rejected key
        # would crash the whole request instead of failing gracefully.
        return {"term": term, "status": "error", "detail": str(exc), "hits": None}

    if not isinstance(results, dict) or "records" not in results:
        return {"term": term, "status": "error", "detail": f"unexpected response: {results}", "hits": None}

    records = results.get("records", [])
    hits = [
        {
            "name": r.get("name"),
            "bucket": r.get("bucket"),
            "date": r.get("date"),
            "media_type": r.get("media"),
            "system_id": r.get("systemid"),
        }
        for r in records
    ]

    return {
        "term": term,
        "status": "ok",
        "hit_count": len(hits),
        "found_on_dark_web": len(hits) > 0,
        "hits": hits,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search dark web / paste sites via Intelligence X")
    parser.add_argument("term", help="Domain, email, or keyword to search for")
    parser.add_argument("--max-results", type=int, default=20)
    args = parser.parse_args()

    api_key = get_intelx_key()
    if not api_key:
        print(json.dumps({
            "status": "error",
            "detail": "Set INTELX_KEY first. Get a free key at https://intelx.io/account?tab=developer",
        }, indent=2))
        sys.exit(1)

    print(json.dumps(search_dark_web(args.term, api_key, args.max_results), indent=2, default=str))
