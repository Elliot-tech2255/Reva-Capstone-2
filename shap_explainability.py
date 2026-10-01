"""
Credential Exposure Monitor -- SHAP Explainability + Evaluation Metrics

Two additions to the ML cross-check (Isolation Forest):

1. SHAP explainability: instead of just "this employee is unusual," this
   shows WHICH of the 5 features drove that verdict, and by how much --
   closing the "black box" gap in the ML step specifically.

2. Proper evaluation metrics: precision, recall, and F1-score, computed
   against the same planted-outlier ground truth used before, presented
   the way an academic evaluation is expected to look.
"""

import json
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import shap

# Reuse the exact same synthetic lab dataset logic as anomaly_detection_check.py
from anomaly_detection_check import build_synthetic_lab_dataset, FEATURES


def run_with_shap(records: list[dict], contamination: float = 0.10):
    X = np.array([[r[f] for f in FEATURES] for r in records])

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    predictions = model.fit_predict(X)  # -1 = outlier, 1 = normal

    # --- SHAP explainability ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # shape: (n_employees, n_features)

    results = []
    for i, r in enumerate(records):
        is_outlier = bool(predictions[i] == -1)
        # Turn raw SHAP values into a simple "% contribution" per feature,
        # only meaningful to show for the ones actually flagged.
        raw = shap_values[i]
        total_abs = np.sum(np.abs(raw)) or 1.0
        contributions = {
            FEATURES[j]: round(float(abs(raw[j]) / total_abs) * 100, 1)
            for j in range(len(FEATURES))
        }
        # sort features by contribution, descending
        top_features = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)

        results.append({
            "employee_id": r["employee_id"],
            "is_outlier": is_outlier,
            "why_flagged": [f"{name} ({pct}%)" for name, pct in top_features] if is_outlier else None,
        })

    return results, predictions


def evaluate(records, predictions, ground_truth_ids):
    y_true = [1 if r["employee_id"] in ground_truth_ids else 0 for r in records]
    y_pred = [1 if p == -1 else 0 for p in predictions]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


if __name__ == "__main__":
    records, ground_truth = build_synthetic_lab_dataset()
    results, predictions = run_with_shap(records)

    print("=== SHAP explanation for every flagged employee ===")
    flagged = [r for r in results if r["is_outlier"]]
    print(json.dumps(flagged, indent=2))

    print("\n=== Evaluation metrics (precision / recall / F1) ===")
    metrics = evaluate(records, predictions, ground_truth)
    print(json.dumps(metrics, indent=2))
