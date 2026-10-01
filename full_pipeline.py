"""
Credential Exposure Monitor -- FULL PIPELINE, END TO END
This is Priya's entire journey through the system, in one script --
every module already built and tested, chained together exactly the way
the architecture diagram describes it.

    Step 1 (Collect)      -> credential_exposure_check.py + user_behavior_analysis.py
    Step 2 (Core Engine)  -> scoring_engine.py + privilege_model.py + anomaly_detection_check.py
    Step 3 (What You See) -> ai_advisor.py

Run this after installing: pip install pycryptodome scikit-learn numpy shap
"""

import json

from scoring_engine import score_employee
from privilege_model import compute_privilege_multiplier
from user_behavior_analysis import build_baseline, score_login
from ai_advisor import get_recommendation


def run_full_check(employee_id: str, signals: dict, access_flags: dict,
                    systems_accessible: int, login_history: list, new_login: dict) -> dict:
    """The complete pipeline for ONE employee, from raw facts to a final
    recommendation -- exactly the flow on your Architecture diagram."""

    # --- Step: figure out this employee's privilege multiplier from real access,
    # not a job title ---
    privilege_result = compute_privilege_multiplier(access_flags, systems_accessible)

    # --- Step: check this specific login against the employee's OWN history ---
    baseline = build_baseline(login_history)
    behavior_result = score_login(baseline, new_login)

    # --- Core Engine: combine the clues (Bayesian) x Impact Severity = Risk score ---
    # scoring_engine expects a privilege LABEL for its built-in multiplier table;
    # here we override with the more precise access-based multiplier we just computed.
    finding = score_employee(employee_id, signals, privilege="regular")  # placeholder call
    likelihood_pct = finding["likelihood_percent"]
    precise_multiplier = privilege_result["privilege_multiplier"]
    final_score = min(100.0, round(likelihood_pct * precise_multiplier, 1))
    finding["impact_multiplier"] = precise_multiplier
    finding["final_risk_score"] = final_score
    finding["severity"] = (
        "Critical" if final_score >= 80 else
        "High" if final_score >= 50 else
        "Medium" if final_score >= 20 else "Low"
    )
    finding["behavior_note"] = "; ".join(behavior_result["flags"])
    finding["behavior_score"] = behavior_result["behavior_score"]

    # --- What You See: AI writes the plain-English fix ---
    advice = get_recommendation(finding, use_real_ai=False)  # switch to True once you have a key

    return {
        "employee_id": employee_id,
        "privilege_detail": privilege_result,
        "behavior_detail": behavior_result,
        "risk_finding": finding,
        "recommendation": advice["recommendation"],
    }


if __name__ == "__main__":
    result = run_full_check(
        employee_id="priya@nexoraretail.com",
        signals={"email_breached": True, "password_reused": True, "dark_web_recent": True},
        access_flags={"admin_rights": True, "customer_data_access": True},
        systems_accessible=8,
        login_history=[
            {"hour": 9, "location": "Bengaluru, India"},
            {"hour": 10, "location": "Bengaluru, India"},
            {"hour": 9, "location": "Bengaluru, India"},
            {"hour": 11, "location": "Bengaluru, India"},
        ],
        new_login={"hour": 3, "location": "Lagos, Nigeria"},
    )

    print("=== FULL PIPELINE RESULT: Priya, end to end ===")
    print(json.dumps(result, indent=2, default=str))
