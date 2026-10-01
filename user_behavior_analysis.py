"""
Credential Exposure Monitor -- User Behavior Analysis (UBA)
A NEW, complementary check suggested by mentor feedback.

WHY THIS IS DIFFERENT FROM THE EXISTING ML STEP
The existing Isolation Forest step compares employees TO EACH OTHER using
static risk facts (breach count, password status, etc). This module compares
each login attempt TO THAT SAME PERSON'S OWN HISTORY -- catching a case the
rest of the pipeline can't: a login that behaves nothing like the real
person, even if their password itself checks out fine.

WHAT IT LOOKS AT
1. Login time -- does this login fall within the hours this person
   normally logs in?
2. Login location -- has this person ever logged in from this
   city/country before?

Both checks build a baseline from the person's OWN past logins -- no
cross-employee comparison, no external data needed.
"""

import json
import numpy as np


def build_baseline(login_history: list[dict]) -> dict:
    """login_history = [{"hour": int, "location": str}, ...] for one employee's
    past logins. Returns their personal normal-behavior baseline."""
    hours = [entry["hour"] for entry in login_history]
    locations = {entry["location"] for entry in login_history}

    return {
        "mean_hour": float(np.mean(hours)),
        "std_hour": float(np.std(hours)) if len(hours) > 1 else 2.0,  # avoid zero-std edge case
        "known_locations": locations,
        "num_logins_seen": len(login_history),
    }


def circular_hour_distance(h1: float, h2: float) -> float:
    """Hours wrap around a 24-hour clock -- 23:00 and 01:00 are only 2 hours
    apart, not 22. This measures the true shortest distance between them."""
    diff = abs(h1 - h2) % 24
    return min(diff, 24 - diff)


def score_login(baseline: dict, new_login: dict) -> dict:
    """new_login = {"hour": int, "location": str}. Flags how much this
    specific login deviates from the employee's own normal pattern."""
    hour_distance = circular_hour_distance(new_login["hour"], baseline["mean_hour"])
    # How many "personal standard deviations" away is this login's time?
    hour_z_score = hour_distance / max(baseline["std_hour"], 0.5)  # floor avoids divide-by-near-zero

    is_new_location = new_login["location"] not in baseline["known_locations"]

    # Simple, explainable combination -- not a black box:
    # time deviation contributes on a 0-60 scale, an unseen location adds a flat 40
    time_component = min(60.0, hour_z_score * 20.0)
    location_component = 40.0 if is_new_location else 0.0
    behavior_score = round(min(100.0, time_component + location_component), 1)

    flags = []
    if hour_z_score > 2.0:
        flags.append(f"Login at {new_login['hour']}:00 is unusual for this person "
                      f"(their typical login time is around {baseline['mean_hour']:.0f}:00)")
    if is_new_location:
        flags.append(f"Never seen a login from '{new_login['location']}' for this person before")

    return {
        "behavior_score": behavior_score,
        "flags": flags if flags else ["Login matches this person's normal pattern"],
        "is_anomalous": behavior_score >= 50.0,
    }


if __name__ == "__main__":
    # Priya's own login history over the past few months -- always logs in
    # during work hours, always from the same city.
    priya_history = [
        {"hour": 9, "location": "Bengaluru, India"},
        {"hour": 10, "location": "Bengaluru, India"},
        {"hour": 9, "location": "Bengaluru, India"},
        {"hour": 11, "location": "Bengaluru, India"},
        {"hour": 9, "location": "Bengaluru, India"},
        {"hour": 10, "location": "Bengaluru, India"},
    ]
    baseline = build_baseline(priya_history)
    print("=== Priya's personal baseline ===")
    print(json.dumps({k: v for k, v in baseline.items() if k != "known_locations"}, indent=2))
    print("Known locations:", baseline["known_locations"])

    print("\n=== Login 1: normal, 10 AM from Bengaluru ===")
    print(json.dumps(score_login(baseline, {"hour": 10, "location": "Bengaluru, India"}), indent=2))

    print("\n=== Login 2: 3 AM from a country never seen before ===")
    print(json.dumps(score_login(baseline, {"hour": 3, "location": "Lagos, Nigeria"}), indent=2))

    print("\n=== Login 3: slightly late (1 PM), still Bengaluru ===")
    print(json.dumps(score_login(baseline, {"hour": 13, "location": "Bengaluru, India"}), indent=2))
