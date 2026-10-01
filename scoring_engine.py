"""
Credential Exposure Monitor -- Step 2: The Core Scoring Engine
This is the "math" step that sits between the raw checks (breach/password/
dark-web lookups) and the ML cross-check / AI advisor.

It does two things:
  1. Combines evidence using Bayesian likelihood ratios -- weak clues barely
     move the needle, strong clues shift it a lot, and clues that AGREE
     compound multiplicatively (not by simple addition).
  2. Multiplies the result by an account-privilege weight, so the same
     evidence on an admin account scores higher than on a regular one.

It also includes a simple password-similarity check -- catching cases like
"Admin@123!" -> "Admin@124!", where an employee technically changed their
password but didn't actually fix anything.
"""

import json

# ---------------------------------------------------------------------------
# Likelihood ratios: how much more often this signal appears on a genuinely
# at-risk account vs. a safe one. These are reasoned starting estimates --
# a production system would recalibrate them against real incident data
# over time, the same way spam filters do.
# ---------------------------------------------------------------------------
LIKELIHOOD_RATIOS = {
    "email_breached": 1.5,        # weak on its own -- breach exposure is common
    "password_reused": 10.6,      # strong -- direct, current exploitability
    "dark_web_recent": 18.3,      # very strong -- rare and highly diagnostic
}

PRIOR_PROBABILITY = 0.05  # baseline: 5% of any given account is genuinely at risk

PRIVILEGE_MULTIPLIER = {
    "regular": 1.0,
    "manager": 1.5,
    "admin": 2.0,
}

SEVERITY_BANDS = [
    (80, "Critical"),
    (50, "High"),
    (20, "Medium"),
    (0, "Low"),
]


def bayesian_likelihood(signals: dict) -> float:
    """Combine evidence using odds multiplication, not simple addition.
    signals = {"email_breached": bool, "password_reused": bool, "dark_web_recent": bool}
    Returns a probability from 0 to 100."""
    prior_odds = PRIOR_PROBABILITY / (1 - PRIOR_PROBABILITY)
    odds = prior_odds
    for signal, is_present in signals.items():
        if is_present and signal in LIKELIHOOD_RATIOS:
            odds *= LIKELIHOOD_RATIOS[signal]
    probability = odds / (1 + odds)
    return round(probability * 100, 1)


def severity_label(score: float) -> str:
    for threshold, label in SEVERITY_BANDS:
        if score >= threshold:
            return label
    return "Low"


def score_employee(employee_id: str, signals: dict, privilege: str) -> dict:
    """Full Core Engine calculation for one employee."""
    likelihood_pct = bayesian_likelihood(signals)
    multiplier = PRIVILEGE_MULTIPLIER.get(privilege, 1.0)
    final_score = min(100.0, round(likelihood_pct * multiplier, 1))

    return {
        "employee_id": employee_id,
        "signals": signals,
        "privilege": privilege,
        "likelihood_percent": likelihood_pct,
        "impact_multiplier": multiplier,
        "final_risk_score": final_score,
        "severity": severity_label(final_score),
    }


def password_similarity_check(old_password: str, new_password: str, max_distance: int = 2) -> dict:
    """Catches the 'Admin@123!' -> 'Admin@124!' problem: a new password that's
    only a tiny edit away from a password already known to be unsafe.
    Uses Levenshtein (edit) distance -- no external library needed."""

    def levenshtein(a: str, b: str) -> int:
        if len(a) < len(b):
            a, b = b, a
        prev_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr_row = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                curr_row.append(min(prev_row[j] + 1, curr_row[j - 1] + 1, prev_row[j - 1] + cost))
            prev_row = curr_row
        return prev_row[-1]

    distance = levenshtein(old_password, new_password)
    too_similar = distance <= max_distance and old_password != new_password

    return {
        "old_password_masked": old_password[:2] + "*" * (len(old_password) - 2),
        "new_password_masked": new_password[:2] + "*" * (len(new_password) - 2),
        "edit_distance": distance,
        "flagged_as_minor_tweak": too_similar,
    }


if __name__ == "__main__":
    print("=== Priya: IT Admin, all three signals present ===")
    priya = score_employee(
        "priya@nexoraretail.com",
        {"email_breached": True, "password_reused": True, "dark_web_recent": True},
        privilege="admin",
    )
    print(json.dumps(priya, indent=2))

    print("\n=== Employee A: one old breach only, safe password, regular staff ===")
    emp_a = score_employee(
        "empA@nexoraretail.com",
        {"email_breached": True, "password_reused": False, "dark_web_recent": False},
        privilege="regular",
    )
    print(json.dumps(emp_a, indent=2))

    print("\n=== Employee B: no old breach, unsafe password + recent dark web hit ===")
    emp_b = score_employee(
        "empB@nexoraretail.com",
        {"email_breached": False, "password_reused": True, "dark_web_recent": True},
        privilege="regular",
    )
    print(json.dumps(emp_b, indent=2))

    print("\n=== Password similarity check: did Employee B actually fix anything? ===")
    check = password_similarity_check("Admin@123!", "Admin@124!")
    print(json.dumps(check, indent=2))
    print("\n(A genuinely different password, for comparison):")
    check2 = password_similarity_check("Admin@123!", "Xk9#mQ2pTrust!")
    print(json.dumps(check2, indent=2))
