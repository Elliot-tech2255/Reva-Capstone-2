"""
Credential Exposure Monitor -- Feedback Loop for Concept Drift
Fixes the "ML sense of normal can drift, no retraining loop" limitation.

Two things happen here, both realistic for a capstone (not a full MLOps
pipeline, but a genuine, working mechanism):

1. The model is retrained on the CURRENT full dataset every single time it
   runs -- "normal" is never based on stale, months-old data, only on
   whoever is in the organization right now.

2. When a security analyst marks a flagged employee as a false positive,
   that feedback is remembered. If false positives keep happening, the
   model's sensitivity (contamination rate) is automatically loosened --
   and a specific pattern that's been cleared before won't re-trigger an
   identical flag.
"""

import json
from sklearn.ensemble import IsolationForest
import numpy as np

from anomaly_detection_check import build_synthetic_lab_dataset, FEATURES

FEEDBACK_LOG_DEFAULT = []  # in a real system this would be a small database table


class DriftAwareDetector:
    """Wraps IsolationForest with analyst feedback so it adapts over time,
    instead of running the exact same fixed assumptions forever."""

    def __init__(self, base_contamination: float = 0.10):
        self.base_contamination = base_contamination
        self.feedback_log = list(FEEDBACK_LOG_DEFAULT)
        self.cleared_patterns = []  # exact feature-vectors an analyst has cleared before

    def _effective_contamination(self) -> float:
        """If analysts keep marking flags as false positives, the model is
        being too sensitive -- loosen it. If nothing's ever been marked a
        false positive, keep the original assumption."""
        false_positives = sum(1 for f in self.feedback_log if f["verdict"] == "false_positive")
        if false_positives >= 3:
            return max(0.03, self.base_contamination * 0.5)  # meaningfully less sensitive
        if false_positives >= 1:
            return max(0.05, self.base_contamination * 0.8)  # slightly less sensitive
        return self.base_contamination

    def record_feedback(self, employee_features: list, verdict: str):
        """verdict = 'false_positive' or 'confirmed_risk'."""
        self.feedback_log.append({"features": employee_features, "verdict": verdict})
        if verdict == "false_positive":
            self.cleared_patterns.append(employee_features)

    def _matches_cleared_pattern(self, features: list, tolerance: float = 0.5) -> bool:
        """If this employee's profile is nearly identical to one an analyst
        already cleared before, don't re-flag the same pattern again."""
        for cleared in self.cleared_patterns:
            if all(abs(a - b) <= tolerance for a, b in zip(features, cleared)):
                return True
        return False

    def run(self, records: list[dict]):
        X = np.array([[r[f] for f in FEATURES] for r in records])

        # Always retrain fresh, on whoever is in the organization right now --
        # this alone prevents comparing today's employees against stale history.
        contamination = self._effective_contamination()
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        predictions = model.fit_predict(X)

        results = []
        for i, r in enumerate(records):
            raw_flag = bool(predictions[i] == -1)
            suppressed = raw_flag and self._matches_cleared_pattern(list(X[i]))
            results.append({
                "employee_id": r["employee_id"],
                "flagged": raw_flag and not suppressed,
                "suppressed_by_past_feedback": suppressed,
            })
        return results, contamination


if __name__ == "__main__":
    records, ground_truth = build_synthetic_lab_dataset()
    detector = DriftAwareDetector(base_contamination=0.10)

    print("=== Run 1: before any feedback ===")
    results, cont = detector.run(records)
    flagged = [r["employee_id"] for r in results if r["flagged"]]
    print(f"contamination used: {cont}")
    print("flagged:", flagged)

    # Simulate an analyst marking one of the flagged employees as a false positive
    # (using a real employee's own feature vector as the "cleared pattern")
    example_employee = next(r for r in records if r["employee_id"] == flagged[0])
    example_features = [example_employee[f] for f in FEATURES]
    print(f"\n=== Analyst marks {flagged[0]} as a FALSE POSITIVE ===")
    detector.record_feedback(example_features, "false_positive")
    detector.record_feedback(example_features, "false_positive")
    detector.record_feedback(example_features, "false_positive")

    print("\n=== Run 2: after 3 false-positive reports ===")
    results, cont = detector.run(records)
    flagged_now = [r["employee_id"] for r in results if r["flagged"]]
    print(f"contamination used: {cont} (loosened from 0.10)")
    print("flagged:", flagged_now)
    print("\nNote: the model is now less sensitive, and the specific pattern")
    print("that was cleared won't re-trigger an identical flag going forward.")
