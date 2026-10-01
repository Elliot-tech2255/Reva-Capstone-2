"""
Credential Exposure Monitor -- Step 3: Anomaly Detection Cross-Check
Unsupervised ML (Isolation Forest), same technique used in Capstone 1.

WHY THIS EXISTS
The Bayesian scoring engine (Step 2) is the primary, explainable decision
maker. This module is a SECOND OPINION only: it looks at every employee's
combined signal pattern at once and flags whoever looks statistically
unusual compared to the rest of the organization -- catching combinations
the fixed Bayesian weights might under-value.

It needs no labeled "who really got hacked" data (which doesn't exist for
this problem) -- Isolation Forest is unsupervised, it only needs to know
what "normal" looks like across the group.

TEST METHOD
Since there's no real company to test against, this uses a synthetic lab
dataset of 40 fake employees (same "Nexora Retail" convention used
throughout the project): 36 built to look ordinary, 4 deliberately planted
as obvious outliers -- a documented answer key, exactly like the lab
environment used for the PQC/TLS modules. A working detector should flag
close to those 4 and few/none of the 36 "normal" ones.
"""

import json
import numpy as np
from sklearn.ensemble import IsolationForest

RNG = np.random.default_rng(42)

FEATURES = ["breach_count", "password_reused", "dark_web_days_ago", "privilege_level", "bayesian_score"]


def build_synthetic_lab_dataset(n_normal: int = 36, n_planted_outliers: int = 4):
    """36 ordinary synthetic employees + 4 deliberately planted high-risk
    outliers. Returns (records, ground_truth_outlier_ids)."""
    records = []

    # -- 36 "normal" employees: low breach count, rarely reused password,
    #    rarely on the dark web recently, mostly regular privilege
    for i in range(n_normal):
        breach_count = RNG.choice([0, 0, 0, 1, 1, 2], p=[0.35, 0.25, 0.15, 0.15, 0.05, 0.05])
        password_reused = RNG.choice([0, 1], p=[0.88, 0.12])
        dark_web_days_ago = int(RNG.choice([999, 999, 999, 999, 300, 500], p=[0.55, 0.15, 0.1, 0.1, 0.05, 0.05]))
        privilege_level = RNG.choice([1, 1, 1, 2], p=[0.7, 0.1, 0.1, 0.1])  # mostly regular
        bayesian_score = float(np.clip(RNG.normal(loc=15, scale=8), 0, 60))
        records.append({
            "employee_id": f"EMP{i+1:03d}",
            "breach_count": int(breach_count),
            "password_reused": int(password_reused),
            "dark_web_days_ago": dark_web_days_ago,
            "privilege_level": int(privilege_level),
            "bayesian_score": round(bayesian_score, 1),
        })

    # -- 4 deliberately planted outliers: multiple strong signals firing
    #    together, high privilege -- the "Employee B" pattern
    planted = [
        {"employee_id": "EMP_PLANT_01", "breach_count": 3, "password_reused": 1,
         "dark_web_days_ago": 20, "privilege_level": 3, "bayesian_score": 94.0},
        {"employee_id": "EMP_PLANT_02", "breach_count": 4, "password_reused": 1,
         "dark_web_days_ago": 45, "privilege_level": 3, "bayesian_score": 91.5},
        {"employee_id": "EMP_PLANT_03", "breach_count": 2, "password_reused": 1,
         "dark_web_days_ago": 10, "privilege_level": 2, "bayesian_score": 88.0},
        {"employee_id": "EMP_PLANT_04", "breach_count": 5, "password_reused": 1,
         "dark_web_days_ago": 5, "privilege_level": 3, "bayesian_score": 99.0},
    ]
    records.extend(planted)
    ground_truth_outliers = {r["employee_id"] for r in planted}
    return records, ground_truth_outliers


def run_anomaly_detection(records: list[dict], contamination: float = 0.10) -> list[dict]:
    """Fit Isolation Forest on the group and flag statistical outliers."""
    X = np.array([[r[f] for f in FEATURES] for r in records])

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    predictions = model.fit_predict(X)          # -1 = outlier, 1 = normal
    scores = model.decision_function(X)          # lower = more anomalous

    results = []
    for r, pred, score in zip(records, predictions, scores):
        results.append({
            "employee_id": r["employee_id"],
            "is_outlier": bool(pred == -1),
            "anomaly_score": round(float(score), 4),
            "bayesian_score": r["bayesian_score"],
        })
    return sorted(results, key=lambda x: x["anomaly_score"])


def validate_against_ground_truth(results: list[dict], ground_truth_outliers: set) -> dict:
    flagged = {r["employee_id"] for r in results if r["is_outlier"]}
    true_positives = flagged & ground_truth_outliers
    false_positives = flagged - ground_truth_outliers
    false_negatives = ground_truth_outliers - flagged
    return {
        "planted_outliers": sorted(ground_truth_outliers),
        "flagged_by_model": sorted(flagged),
        "correctly_caught": sorted(true_positives),
        "missed": sorted(false_negatives),
        "false_alarms_on_normal_employees": sorted(false_positives),
        "detection_rate": f"{len(true_positives)}/{len(ground_truth_outliers)}",
    }


if __name__ == "__main__":
    records, ground_truth = build_synthetic_lab_dataset()
    results = run_anomaly_detection(records)
    validation = validate_against_ground_truth(results, ground_truth)

    print("=== Top 6 most anomalous employees (lowest anomaly_score = most unusual) ===")
    print(json.dumps(results[:6], indent=2))
    print("\n=== Validation against the planted ground-truth answer key ===")
    print(json.dumps(validation, indent=2))
