"""
Credential Exposure Monitor -- Granular Privilege Scoring
Fixes the "only 3 tiers" limitation: instead of a single job-title category
(regular / manager / admin), this scores privilege based on WHAT an account
can actually reach -- so a finance employee with no "admin" title, but real
access to money-moving systems, correctly scores as high-privilege too.
"""

import json

# Each access flag contributes its own weight -- these stack, so an account
# with multiple sensitive access rights scores higher than one with just one.
ACCESS_WEIGHTS = {
    "admin_rights": 0.50,          # can administer systems/accounts
    "financial_access": 0.40,      # can view/move money, payroll, banking
    "customer_data_access": 0.30,  # can view customer PII
    "source_code_access": 0.20,    # can view/modify production code
    "hr_data_access": 0.25,        # can view employee personal data
}

SYSTEMS_ACCESS_STEP = 0.05  # small extra weight per distinct system reachable
SYSTEMS_ACCESS_CAP = 0.20   # don't let raw system count alone dominate the score

MIN_MULTIPLIER = 1.0
MAX_MULTIPLIER = 2.0


def compute_privilege_multiplier(access_flags: dict, systems_accessible: int = 0) -> dict:
    """access_flags = {"admin_rights": bool, "financial_access": bool, ...}
    Returns the same 1.0-2.0 multiplier scoring_engine.py already expects,
    but now driven by actual access rather than a job-title guess."""

    raw_score = sum(weight for flag, weight in ACCESS_WEIGHTS.items() if access_flags.get(flag))
    systems_bonus = min(systems_accessible * SYSTEMS_ACCESS_STEP, SYSTEMS_ACCESS_CAP)
    raw_score += systems_bonus

    multiplier = min(MAX_MULTIPLIER, MIN_MULTIPLIER + raw_score)

    contributing = [flag for flag in ACCESS_WEIGHTS if access_flags.get(flag)]
    return {
        "raw_access_score": round(raw_score, 2),
        "privilege_multiplier": round(multiplier, 2),
        "contributing_access_rights": contributing,
        "systems_accessible": systems_accessible,
    }


if __name__ == "__main__":
    print("=== Regular employee, no special access ===")
    print(json.dumps(compute_privilege_multiplier({}, systems_accessible=1), indent=2))

    print("\n=== 'Finance analyst' -- no admin title, but real financial + HR access ===")
    print(json.dumps(compute_privilege_multiplier(
        {"financial_access": True, "hr_data_access": True}, systems_accessible=3
    ), indent=2))

    print("\n=== IT Admin -- full admin rights, broad system reach ===")
    print(json.dumps(compute_privilege_multiplier(
        {"admin_rights": True, "customer_data_access": True, "source_code_access": True},
        systems_accessible=8,
    ), indent=2))

    print("\n=== CFO -- financial + HR + customer data, but no 'admin' rights at all ===")
    print(json.dumps(compute_privilege_multiplier(
        {"financial_access": True, "hr_data_access": True, "customer_data_access": True},
        systems_accessible=4,
    ), indent=2))
