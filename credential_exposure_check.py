"""
PQC... wait, wrong project -- Credential Exposure Monitor
Checking Layer: Breach + Paste Exposure + Password Reuse Checker
Data source: XposedOrNot (https://xposedornot.com) -- free, open, no API key
required for these endpoints, explicitly built for developer integration.

This module replaces the Ahmia-based design: Ahmia's robots.txt blocks
automated access, but XposedOrNot is a REST API meant to be queried
programmatically, and its free tier covers everything the "Checking Layer"
needs: email breach lookup, paste-site exposure (the same "hacker forum /
paste dump" ground Ahmia would have covered), and privacy-safe password
reuse checking.

Usage:
    python3 credential_exposure_check.py j.doe@nexoraretail.com --password "hunter2"
"""

import argparse
import json
import time
import urllib.request
import urllib.error

from Crypto.Hash import keccak

BASE = "https://api.xposedornot.com/v1"
PASS_BASE = "https://passwords.xposedornot.com/api/v1/pass/anon"


def _get(url: str) -> dict:
    """GET a URL and return parsed JSON, respecting XposedOrNot's free-tier
    rate limit of 2 requests/second by adding a small delay.

    Returns a dict that always has a "status" key: "ok", "not_found", or
    "error" -- callers must check this rather than assuming an empty/odd
    response means "nothing was found". A blocked or failed request must
    never be reported to a user as "this password/email is safe".
    """
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-credential-monitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode())
            body["_status"] = "ok"
            return body
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"_status": "not_found"}
        if e.code == 429:
            return {"_status": "error", "_detail": "rate_limited: free tier is 2 req/s, 25/hr, 100/day"}
        return {"_status": "error", "_detail": f"http_{e.code}: {e.reason}"}
    except (urllib.error.URLError, TimeoutError) as e:
        return {"_status": "error", "_detail": f"network_error: {e}"}
    finally:
        time.sleep(0.55)  # stay safely under 2 requests/second


def check_email_breaches(email: str) -> dict:
    """Which named breaches has this email appeared in?"""
    data = _get(f"{BASE}/check-email/{email}")
    if data["_status"] == "error":
        return {"email": email, "status": "error", "detail": data["_detail"]}
    if data["_status"] == "not_found":
        return {"email": email, "status": "ok", "breached": False, "breaches": []}
    breaches = data.get("breaches", [[]])[0] if data.get("breaches") else []
    return {"email": email, "status": "ok", "breached": len(breaches) > 0, "breaches": breaches}


def breach_analytics(email: str) -> dict:
    """Full detail: what data was exposed, how risky, and whether the email
    turned up on any paste site (Pastebin-style dumps -- the same territory
    a dark-web paste monitor would cover)."""
    data = _get(f"{BASE}/breach-analytics?email={email}")
    if data["_status"] == "error":
        return {"email": email, "status": "error", "detail": data["_detail"]}

    exposed = data.get("ExposedBreaches")
    metrics = data.get("BreachMetrics") or {}
    pastes = data.get("PastesSummary") or {}

    risk_label, risk_score = "Unknown", None
    if metrics.get("risk"):
        risk_label = metrics["risk"][0].get("risk_label", "Unknown")
        risk_score = metrics["risk"][0].get("risk_score")

    # Password storage safety across all this email's breaches -- e.g. was
    # the password ever stored in plain text vs. properly hashed.
    password_strength = {"EasyToCrack": 0, "PlainText": 0, "StrongHash": 0, "Unknown": 0}
    if metrics.get("passwords_strength"):
        password_strength.update(metrics["passwords_strength"][0])

    # What TYPES of data got exposed (emails, passwords, phone numbers, etc.),
    # flattened from XposedOrNot's nested category tree into a simple count
    # per leaf data type -- same substance as their "What Data Was Exposed"
    # view, just flattened for a simpler chart.
    exposed_data_counts = {}
    for category in (metrics.get("xposed_data") or [{}])[0].get("children", []):
        for leaf in category.get("children", []):
            name = leaf.get("name", "").replace("data_", "")
            exposed_data_counts[name] = exposed_data_counts.get(name, 0) + leaf.get("value", 1)

    return {
        "email": email,
        "status": "ok",
        "found": exposed is not None,
        "risk_label": risk_label,
        "risk_score": risk_score,
        "breach_count": len(exposed["breaches_details"]) if exposed else 0,
        "breach_details": exposed["breaches_details"] if exposed else [],
        "password_strength": password_strength,
        "exposed_data_types": exposed_data_counts,
        "paste_exposure_count": pastes.get("cnt", 0),
        "paste_last_seen": pastes.get("tmpstmp") or None,
    }


def check_password_exposed(password: str) -> dict:
    """Privacy-safe password check: hash locally with Keccak-512, send only
    the first 10 hex characters. The real password never leaves this machine.

    IMPORTANT: on a network failure or error this returns status="error" and
    exposed=None -- never False. Treating "we couldn't check" the same as
    "confirmed safe" would be a dangerous false negative in a security tool.
    """
    h = keccak.new(digest_bits=512)
    h.update(password.encode("utf-8"))
    prefix = h.hexdigest()[:10]

    data = _get(f"{PASS_BASE}/{prefix}")
    if data["_status"] == "error":
        return {"status": "error", "detail": data["_detail"], "exposed": None}
    if data["_status"] == "not_found":
        return {"status": "ok", "exposed": False, "times_seen": 0}

    result = data.get("SearchPassAnon")
    if not result:
        return {"status": "ok", "exposed": False, "times_seen": 0}
    return {
        "status": "ok",
        "exposed": True,
        "times_seen": int(result.get("count", 0)),
        "composition": result.get("char"),   # e.g. "D:6;A:0;S:0;L:6"
        "in_common_wordlist": bool(result.get("wordlist")),
    }


def full_check(email: str, password: str | None = None) -> dict:
    """Run everything the Checking Layer needs for one employee account."""
    result = {
        "email": email,
        "breach_summary": check_email_breaches(email),
        "breach_details": breach_analytics(email),
    }
    if password:
        result["password_check"] = check_password_exposed(password)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check an email/password against XposedOrNot")
    parser.add_argument("email", help="Employee email address to check")
    parser.add_argument("--password", help="Password to check for reuse (never transmitted in full)")
    args = parser.parse_args()

    print(json.dumps(full_check(args.email, args.password), indent=2, default=str))
