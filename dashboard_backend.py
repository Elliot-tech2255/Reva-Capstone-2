"""
Credential Exposure Monitor -- Dashboard Backend (Sprint 6)
Wraps the existing, already-tested pipeline (full_pipeline.py) in a real
FastAPI + SQLite backend, so results can be browsed on a dashboard instead
of just read as JSON in a terminal.

This file does NOT change the logic in any of the 9 existing modules --
it only imports and calls them, exactly per the plan.

Run with:
    uvicorn dashboard_backend:app --reload --port 8000
Then open dashboard.html in a browser (it calls http://localhost:8000).
"""

import json
import sqlite3
from contextlib import contextmanager
from io import StringIO
import csv as csv_module

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from full_pipeline import run_full_check
from credential_exposure_check import check_email_breaches, breach_analytics, check_password_exposed
from privilege_model import compute_privilege_multiplier
from scoring_engine import bayesian_likelihood, severity_label
from ai_advisor import get_recommendation
from user_behavior_analysis import build_baseline, score_login
import google_sso
import secrets
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from darkweb_paste_search import search_dark_web, get_intelx_key
from sklearn.ensemble import IsolationForest
import shap
import os

DB_PATH = "credential_monitor.db"

app = FastAPI(title="Credential Exposure Risk Monitor")

print("=" * 60)
print("GOOGLE SSO STARTUP CHECK (this process, right now):")
print("  GOOGLE_CLIENT_ID set:", bool(google_sso.GOOGLE_CLIENT_ID))
print("  GOOGLE_CLIENT_SECRET set:", bool(google_sso.GOOGLE_CLIENT_SECRET))
print("  is_configured():", google_sso.is_configured())
print("=" * 60)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                employee_id TEXT PRIMARY KEY,
                risk_score REAL,
                severity TEXT,
                signals TEXT,
                privilege_detail TEXT,
                behavior_detail TEXT,
                recommendation TEXT,
                data_source TEXT DEFAULT 'synthetic_demo',
                password_checked INTEGER DEFAULT 1,
                behavior_checked INTEGER DEFAULT 1,
                access_checked INTEGER DEFAULT 1,
                ml_flagged INTEGER DEFAULT 0,
                ml_reason TEXT DEFAULT '',
                ml_contributions_json TEXT DEFAULT '[]',
                breach_detail_json TEXT DEFAULT '[]',
                password_strength_json TEXT DEFAULT '{}',
                exposed_data_json TEXT DEFAULT '{}',
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Never store the actual password or its hash -- only pass/fail
        # signals, consistent with the privacy design in the report.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                verdict TEXT,
                feature_snapshot TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Real login events from actual Google sign-ins, accumulating over
        # time -- this is the REAL baseline data source, replacing the
        # manually-typed "typical hour/location" simulator once enough
        # genuine logins exist for a given employee.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS login_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                login_hour INTEGER,
                login_location TEXT,
                ip_address TEXT,
                lat REAL,
                lon REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Server-side session tracking -- the cookie alone only tells the
        # EMPLOYEE'S OWN browser whether they're signed in. This table is
        # what lets an ANALYST see, for any employee, when they signed in,
        # whether that session is still active right now, and when (or
        # whether) they signed out.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                login_at TEXT DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT,
                logout_at TEXT,
                login_location TEXT
            )
        """)
        conn.commit()

        # Automatic, non-destructive schema migration: if a column this
        # code expects is missing from an existing database (because it
        # was created before that column was added), add it safely rather
        # than requiring the whole file to be deleted. Every existing row
        # and every column already present is left completely untouched --
        # this only ever ADDS columns, never removes or rewrites data. This
        # is what lets the database persist across every future restart,
        # not just today's.
        expected_columns = {
            "findings": [
                ("data_source", "TEXT DEFAULT 'synthetic_demo'"),
                ("password_checked", "INTEGER DEFAULT 1"),
                ("password_reset_required", "INTEGER DEFAULT 0"),
                ("behavior_checked", "INTEGER DEFAULT 1"),
                ("access_checked", "INTEGER DEFAULT 1"),
                ("ml_flagged", "INTEGER DEFAULT 0"),
                ("ml_reason", "TEXT DEFAULT ''"),
                ("ml_contributions_json", "TEXT DEFAULT '[]'"),
                ("breach_detail_json", "TEXT DEFAULT '[]'"),
                ("password_strength_json", "TEXT DEFAULT '{}'"),
                ("exposed_data_json", "TEXT DEFAULT '{}'"),
                ("updated_at", "TEXT"),  # SQLite forbids CURRENT_TIMESTAMP as an ALTER-added default; backfilled below instead
            ],
            "sessions": [
                ("login_location", "TEXT"),
            ],
            "login_events": [
                ("lat", "REAL"),
                ("lon", "REAL"),
            ],
        }
        for table, columns in expected_columns.items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col_name, col_def in columns:
                if col_name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
        conn.commit()
        # Backfill updated_at for any rows that predate the column existing
        conn.execute("UPDATE findings SET updated_at = CURRENT_TIMESTAMP WHERE updated_at IS NULL")
        conn.commit()


# ---------------------------------------------------------------------------
# Synthetic lab dataset -- same "36 normal + 4 planted outliers" convention
# used throughout the project, now extended with the fields the FULL
# pipeline needs (access flags, login history), not just the 5 ML features.
# ---------------------------------------------------------------------------

LOCATIONS_NORMAL = ["Bengaluru, India", "Mumbai, India", "Chennai, India"]
LOCATIONS_RARE = ["Lagos, Nigeria", "Bucharest, Romania", "Manila, Philippines"]


def _make_login_history(rng, normal_location: str, n=6) -> list[dict]:
    return [{"hour": int(rng.integers(9, 12)), "location": normal_location} for _ in range(n)]


def build_dataset():
    # Re-seed fresh on every call -- the server stays running between clicks,
    # so a module-level RNG would keep drawing further along the same
    # sequence each time, making "Run assessment" produce different results
    # on every click instead of the same reproducible synthetic dataset.
    rng = np.random.default_rng(7)
    employees = []

    # Signal templates chosen so each band is reachable, verified against the
    # REAL bayesian_likelihood/compute_privilege_multiplier functions below
    # (rejection sampling) rather than hand-calculated and hoped -- this
    # guarantees roughly 19 employees land in each of Low/Medium/High/Critical,
    # not left to chance the way independent random probabilities would skew
    # everything toward Low.
    band_templates = {
        "Low": {"email_breached": False, "password_reused": False, "dark_web_recent": False},
        "Medium": {"email_breached": False, "password_reused": True, "dark_web_recent": False},
        "High": {"email_breached": True, "password_reused": True, "dark_web_recent": False},
        "Critical": {"email_breached": False, "password_reused": True, "dark_web_recent": True},
    }
    band_ranges = {"Low": (0, 20), "Medium": (20, 50), "High": (50, 80), "Critical": (80, 101)}

    emp_num = 1
    for band, target_signals in band_templates.items():
        lo, hi = band_ranges[band]
        placed = 0
        attempts = 0
        while placed < 19 and attempts < 500:
            attempts += 1
            home = rng.choice(LOCATIONS_NORMAL)
            access_flags = {
                "admin_rights": bool(rng.random() > 0.85),
                "financial_access": bool(rng.random() > 0.85),
                "customer_data_access": bool(rng.random() > 0.6),
            }
            systems = int(rng.integers(1, 6))
            priv = compute_privilege_multiplier(access_flags, systems)
            score = min(100.0, round(bayesian_likelihood(target_signals) * priv["privilege_multiplier"], 1))
            if not (lo <= score < hi):
                continue  # this random access combo pushed the score out of the target band -- retry
            # ~15% of employees get an anomalous login instead of always a
            # normal one -- without this, every random employee shows
            # identical "matches normal pattern" behavior, which is dull for
            # a demo and doesn't showcase the behavior-check feature at all.
            if rng.random() < 0.15:
                new_login = {"hour": int(rng.integers(0, 5)), "location": rng.choice(LOCATIONS_RARE)}
            else:
                new_login = {"hour": int(rng.integers(9, 12)), "location": home}

            employees.append({
                "employee_id": f"emp{emp_num:03d}@nexoraretail.com",
                "signals": dict(target_signals),
                "access_flags": access_flags,
                "systems_accessible": systems,
                "login_history": _make_login_history(rng, home),
                "new_login": new_login,
            })
            emp_num += 1
            placed += 1

    planted = [
        {
            "employee_id": "priya@nexoraretail.com",
            "signals": {"email_breached": True, "password_reused": True, "dark_web_recent": True},
            "access_flags": {"admin_rights": True, "customer_data_access": True},
            "systems_accessible": 8,
            "login_history": _make_login_history(rng, "Bengaluru, India"),
            "new_login": {"hour": 3, "location": "Lagos, Nigeria"},
        },
        {
            "employee_id": "rahul_finance@nexoraretail.com",
            "signals": {"email_breached": True, "password_reused": True, "dark_web_recent": False},
            "access_flags": {"financial_access": True, "hr_data_access": True},
            "systems_accessible": 4,
            "login_history": _make_login_history(rng, "Mumbai, India"),
            "new_login": {"hour": 10, "location": "Mumbai, India"},
        },
        {
            "employee_id": "amit_it@nexoraretail.com",
            "signals": {"email_breached": False, "password_reused": True, "dark_web_recent": True},
            "access_flags": {"admin_rights": True, "source_code_access": True},
            "systems_accessible": 6,
            "login_history": _make_login_history(rng, "Chennai, India"),
            "new_login": {"hour": 2, "location": "Bucharest, Romania"},
        },
        {
            "employee_id": "sara_hr@nexoraretail.com",
            "signals": {"email_breached": True, "password_reused": False, "dark_web_recent": True},
            "access_flags": {"hr_data_access": True, "customer_data_access": True},
            "systems_accessible": 3,
            "login_history": _make_login_history(rng, "Bengaluru, India"),
            "new_login": {"hour": 23, "location": "Manila, Philippines"},
        },
    ]
    employees.extend(planted)
    return employees


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    init_db()


@app.post("/api/run-check/{employee_id}")
def run_check(employee_id: str):
    dataset = build_dataset()
    record = next((e for e in dataset if e["employee_id"] == employee_id), None)
    if not record:
        raise HTTPException(404, f"No synthetic data for {employee_id}")

    result = run_full_check(
        employee_id=record["employee_id"],
        signals=record["signals"],
        access_flags=record["access_flags"],
        systems_accessible=record["systems_accessible"],
        login_history=record["login_history"],
        new_login=record["new_login"],
    )

    with get_db() as conn:
        conn.execute(
            """INSERT INTO findings (employee_id, risk_score, severity, signals,
               privilege_detail, behavior_detail, recommendation,
               data_source, password_checked, behavior_checked, access_checked)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(employee_id) DO UPDATE SET
               risk_score=excluded.risk_score, severity=excluded.severity,
               signals=excluded.signals, privilege_detail=excluded.privilege_detail,
               behavior_detail=excluded.behavior_detail, recommendation=excluded.recommendation,
               data_source=excluded.data_source, password_checked=excluded.password_checked,
               behavior_checked=excluded.behavior_checked, access_checked=excluded.access_checked,
               updated_at=CURRENT_TIMESTAMP""",
            (
                result["employee_id"],
                result["risk_finding"]["final_risk_score"],
                result["risk_finding"]["severity"],
                json.dumps(result["risk_finding"]["signals"]),
                json.dumps(result["privilege_detail"]),
                json.dumps(result["behavior_detail"]),
                result["recommendation"],
                "synthetic_demo", 1, 1, 1,
            ),
        )
        conn.commit()
    return result


@app.post("/api/run-all")
def run_all():
    dataset = build_dataset()
    for record in dataset:
        run_check(record["employee_id"])
    seed_synthetic_sessions()
    ml_result = recompute_ml_flags()
    return {"status": "ok", "checked": len(dataset), "ml_cross_check": ml_result}


def seed_synthetic_sessions():
    """Gives a handful of synthetic employees real-looking session data, so
    the demo dataset showcases the session/concurrent-login features too --
    not just the behavior-pattern ones. Deterministic (fixed employee IDs),
    so it's reproducible run to run, same principle as the rest of the
    synthetic generator."""
    now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    expires = (now + timedelta(hours=5)).isoformat()

    # Real approximate coordinates for each named city -- needed so the
    # map has actual points to plot, not just text labels in the session
    # table (which is a separate data path the map doesn't read from).
    CITY_COORDS = {
        "Bengaluru, India": (12.9716, 77.5946),
        "Lagos, Nigeria": (6.5244, 3.3792),
        "Mumbai, India": (19.0760, 72.8777),
        "Moscow, Russia": (55.7558, 37.6173),
        "Chennai, India": (13.0827, 80.2707),
        "Hyderabad, India": (17.3850, 78.4867),
        "Pune, India": (18.5204, 73.8567),
    }

    with get_db() as conn:
        # Two employees get a genuinely suspicious concurrent session --
        # same detection path real SSO activity would trigger, escalated
        # to Critical for the same reason.
        for emp_id, loc_a, loc_b in [
            ("emp005@nexoraretail.com", "Bengaluru, India", "Lagos, Nigeria"),
            ("emp022@nexoraretail.com", "Mumbai, India", "Moscow, Russia"),
        ]:
            login_a = now.isoformat(sep=" ", timespec="seconds")
            conn.execute(
                "INSERT INTO sessions (employee_id, login_at, expires_at, login_location) VALUES (?, ?, ?, ?)",
                (emp_id, login_a, expires, loc_a),
            )
            lat_a, lon_a = CITY_COORDS[loc_a]
            conn.execute(
                "INSERT INTO login_events (employee_id, login_hour, login_location, ip_address, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
                (emp_id, now.hour, loc_a, "synthetic", lat_a, lon_a),
            )

            login_b = (now + timedelta(minutes=3)).isoformat(sep=" ", timespec="seconds")
            conn.execute(
                "INSERT INTO sessions (employee_id, login_at, expires_at, login_location) VALUES (?, ?, ?, ?)",
                (emp_id, login_b, expires, loc_b),
            )
            lat_b, lon_b = CITY_COORDS[loc_b]
            conn.execute(
                "INSERT INTO login_events (employee_id, login_hour, login_location, ip_address, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
                (emp_id, now.hour, loc_b, "synthetic", lat_b, lon_b),
            )

            row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (emp_id,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE findings SET risk_score = 100.0, severity = 'Critical', "
                    "recommendation = ?, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                    (
                        f"URGENT: Concurrent access detected \u2014 active sessions from both '{loc_a}' and "
                        f"'{loc_b}' at nearly the same time. Lock this account and contact the employee immediately.",
                        emp_id,
                    ),
                )

        # A few more get a single, normal active session -- showing what a
        # clean, non-suspicious "Active session" state looks like too.
        for emp_id, loc in [
            ("emp010@nexoraretail.com", "Chennai, India"),
            ("emp031@nexoraretail.com", "Hyderabad, India"),
            ("emp048@nexoraretail.com", "Pune, India"),
        ]:
            conn.execute(
                "INSERT INTO sessions (employee_id, login_at, expires_at, login_location) VALUES (?, ?, ?, ?)",
                (emp_id, now.isoformat(sep=" ", timespec="seconds"), expires, loc),
            )
            lat, lon = CITY_COORDS[loc]
            conn.execute(
                "INSERT INTO login_events (employee_id, login_hour, login_location, ip_address, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
                (emp_id, now.hour, loc, "synthetic", lat, lon),
            )

        conn.commit()


def recompute_ml_flags():
    """The ML cross-check, actually wired to the real population this time.
    Pulls EVERY current finding (synthetic demo + real uploads together),
    builds a feature vector, and runs Isolation Forest across all of them
    at once -- so a newly uploaded real employee gets compared against
    whoever else is already in the system, exactly as designed.

    Also honors accumulated analyst feedback: repeat false-positive reports
    loosen sensitivity, and patterns an analyst has specifically cleared
    before won't re-trigger an identical flag -- same logic verified in
    feedback_loop.py, now backed by a persistent table instead of memory.
    """
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM findings").fetchall()
        feedback_rows = conn.execute("SELECT * FROM feedback WHERE verdict = 'false_positive'").fetchall()

    if len(rows) < 10:
        return {"status": "skipped", "reason": f"only {len(rows)} employees; need at least 10"}

    FEATURES = ["email_breached", "password_reused", "dark_web_recent", "privilege_multiplier", "risk_score"]

    def feature_vector(signals, priv, risk_score):
        return [
            1 if signals.get("email_breached") else 0,
            1 if signals.get("password_reused") else 0,
            1 if signals.get("dark_web_recent") else 0,
            priv.get("privilege_multiplier", 1.0),
            risk_score,
        ]

    ids, X = [], []
    for r in rows:
        signals = json.loads(r["signals"])
        priv = json.loads(r["privilege_detail"])
        ids.append(r["employee_id"])
        X.append(feature_vector(signals, priv, r["risk_score"]))
    X = np.array(X)

    # Same tiered adjustment as feedback_loop.py: more repeat false positives
    # -> loosen sensitivity. Zero reports keeps the original assumption.
    fp_count = len(feedback_rows)
    if fp_count >= 3:
        contamination = 0.05
    elif fp_count >= 1:
        contamination = 0.08
    else:
        contamination = 0.10

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    predictions = model.fit_predict(X)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    cleared_patterns = [np.array(json.loads(f["feature_snapshot"])) for f in feedback_rows]

    def matches_cleared_pattern(vec, tolerance=0.5):
        return any(np.all(np.abs(vec - cp) <= tolerance) for cp in cleared_patterns)

    with get_db() as conn:
        for i, employee_id in enumerate(ids):
            raw_flag = bool(predictions[i] == -1)
            suppressed = raw_flag and matches_cleared_pattern(X[i])
            is_flagged = raw_flag and not suppressed

            reason = ""
            if is_flagged:
                raw = shap_values[i]
                total_abs = np.sum(np.abs(raw)) or 1.0
                contributions = sorted(
                    zip(FEATURES, [abs(v) / total_abs * 100 for v in raw]),
                    key=lambda kv: kv[1], reverse=True,
                )
                top = contributions[0]
                reason = f"{top[0].replace('_', ' ')} ({top[1]:.0f}% contribution) drove this flag"
                contributions_list = [{"feature": f.replace("_", " "), "pct": round(p, 1)} for f, p in contributions]
            elif suppressed:
                reason = "Previously marked as a false positive by an analyst"
                contributions_list = []
            else:
                contributions_list = []

            conn.execute(
                "UPDATE findings SET ml_flagged = ?, ml_reason = ?, ml_contributions_json = ? WHERE employee_id = ?",
                (1 if is_flagged else 0, reason, json.dumps(contributions_list), employee_id),
            )
        conn.commit()

    return {
        "status": "ok", "population_size": len(rows), "flagged": int(sum(predictions == -1)),
        "contamination_used": contamination, "false_positive_reports": fp_count,
    }


@app.post("/api/force-password-reset/{employee_id}")
def force_password_reset(employee_id: str):
    """Admin flags this account as requiring a new password -- same
    pattern as Okta/Google Workspace/Azure AD's own 'force password
    reset' button. The admin NEVER sets or sees the actual value; only
    the employee submitting their own new password through the password
    portal (reset_password below) can clear this flag. This keeps the
    admin dashboard from ever handling real credentials, while still
    giving a genuine, meaningful lockout-style action for a compromised
    account."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (employee_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"No finding for {employee_id}")
        conn.execute(
            "UPDATE findings SET password_reset_required = 1, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
            (employee_id,),
        )
        conn.commit()
    return {"status": "ok", "employee_id": employee_id, "password_reset_required": True}



@app.post("/api/force-logout/{employee_id}")
def force_logout(employee_id: str):
    """Admin action to immediately terminate every active session for this
    employee -- the real answer to 'get a compromised account logged out
    now', without ever touching or asking for a password. This is what a
    real admin console (Google Workspace, Okta, etc.) offers for exactly
    this scenario, and it's applicable to any employee, not just one."""
    logout_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    with get_db() as conn:
        result = conn.execute(
            "UPDATE sessions SET logout_at = ? WHERE employee_id = ? AND logout_at IS NULL",
            (logout_time, employee_id),
        )
        terminated = result.rowcount
        conn.commit()
    return {"status": "ok", "employee_id": employee_id, "sessions_terminated": terminated, "logout_at": logout_time}



@app.post("/api/feedback/{employee_id}")
def submit_feedback(employee_id: str, verdict: str):
    """verdict = 'false_positive' or 'confirmed_risk'. Records the employee's
    CURRENT feature snapshot as the pattern to recognize/suppress in future
    ML comparisons, then immediately re-runs the cross-check."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (employee_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"No finding for {employee_id}")

        signals = json.loads(row["signals"])
        priv = json.loads(row["privilege_detail"])
        snapshot = [
            1 if signals.get("email_breached") else 0,
            1 if signals.get("password_reused") else 0,
            1 if signals.get("dark_web_recent") else 0,
            priv.get("privilege_multiplier", 1.0),
            row["risk_score"],
        ]
        conn.execute(
            "INSERT INTO feedback (employee_id, verdict, feature_snapshot) VALUES (?, ?, ?)",
            (employee_id, verdict, json.dumps(snapshot)),
        )
        conn.commit()

    ml_result = recompute_ml_flags()
    return {"status": "ok", "employee_id": employee_id, "verdict": verdict, "ml_cross_check": ml_result}


@app.post("/api/simulate-concurrent-login")
async def simulate_concurrent_login(employee_id: str = Form(...), location: str = Form(...)):
    """For reliably demonstrating concurrent-session detection live (e.g.
    in a viva) without depending on two physical devices coincidentally
    reporting different IP-based locations -- which they often won't, especially
    over a USB tunnel where both devices appear to come from the same
    address. This runs the EXACT same detection query the real OAuth
    callback uses; only the location is presenter-supplied instead of
    IP-derived, so the underlying logic being demonstrated is completely real."""
    now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    login_at_ist = now.isoformat(sep=" ", timespec="seconds")
    expires_at = (now + timedelta(hours=5)).isoformat()

    with get_db() as conn:
        active_elsewhere = conn.execute(
            "SELECT login_location FROM sessions WHERE employee_id = ? AND logout_at IS NULL AND expires_at > ? AND login_location != ? AND login_location IS NOT NULL",
            (employee_id, login_at_ist, location),
        ).fetchall()

        concurrent_warning = None
        if active_elsewhere:
            other_locations = ", ".join(sorted(set(r["login_location"] for r in active_elsewhere)))
            concurrent_warning = f"CONCURRENT ACCESS DETECTED: an active session already exists from '{other_locations}' while this new one is from '{location}' \u2014 possible account compromise."

        conn.execute(
            "INSERT INTO sessions (employee_id, login_at, expires_at, login_location) VALUES (?, ?, ?, ?)",
            (employee_id, login_at_ist, expires_at, location),
        )
        conn.commit()

        if concurrent_warning:
            row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (employee_id,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE findings SET risk_score = 100.0, severity = 'Critical', "
                    "recommendation = ?, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                    (f"URGENT: {concurrent_warning} Lock this account and force a password reset immediately.", employee_id),
                )
                conn.commit()
                recompute_ml_flags()

    return {"status": "ok", "concurrent_detected": bool(concurrent_warning), "warning": concurrent_warning, "location": location}


@app.post("/api/test-login-behavior")
async def test_login_behavior(
    employee_id: str = Form(...),
    typical_hour: int = Form(...),
    typical_location: str = Form(...),
    new_hour: int = Form(...),
    new_location: str = Form(...),
):
    """Lets an analyst simulate a login for a specific employee and see
    whether it matches their normal pattern -- the real, previously-wired-
    to-nothing user_behavior_analysis.py logic, now actually reachable.
    A real deployment would build the baseline from actual SSO/AD login
    logs over time; this lets you test the same detection logic today
    without waiting on that integration."""
    baseline_history = [{"hour": typical_hour, "location": typical_location} for _ in range(6)]
    baseline = build_baseline(baseline_history)
    behavior_result = score_login(baseline, {"hour": new_hour, "location": new_location})

    with get_db() as conn:
        row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (employee_id,)).fetchone()
        if not row:
            raise HTTPException(404, f"No finding for {employee_id} yet -- add them first")

        signals = json.loads(row["signals"])
        priv = json.loads(row["privilege_detail"])
        likelihood_pct = bayesian_likelihood(signals)
        final_score = min(100.0, round(likelihood_pct * priv.get("privilege_multiplier", 1.0), 1))
        severity = severity_label(final_score)

        finding_for_ai = {
            "employee_id": employee_id, "signals": signals, "privilege": "regular",
            "final_risk_score": final_score, "severity": severity,
            "behavior_note": "; ".join(behavior_result["flags"]),
        }
        ai_result = get_recommendation(finding_for_ai, use_real_ai=False)

        conn.execute(
            """UPDATE findings SET behavior_detail = ?, behavior_checked = 1,
               recommendation = ?, updated_at = CURRENT_TIMESTAMP
               WHERE employee_id = ?""",
            (json.dumps(behavior_result), ai_result["recommendation"], employee_id),
        )
        conn.commit()

    recompute_ml_flags()
    return {"status": "ok", "employee_id": employee_id, "behavior_result": behavior_result}


@app.post("/api/reset-password")
async def reset_password(email: str = Form(...), password: str = Form(...)):
    """The ONLY place in this system a real password is ever handled.
    It's hashed via k-anonymity immediately (credential_exposure_check.py),
    and the raw value is never logged, stored, or written to the database --
    only the pass/fail result and the resulting risk score are kept."""
    result = check_password_exposed(password)
    password = None  # drop the reference the moment it's no longer needed

    if result.get("status") == "error":
        return {"status": "error", "message": "Could not check your password right now. Please try again shortly."}

    is_unsafe = bool(result["exposed"])

    with get_db() as conn:
        # Correlate with SSO activity: a password check submitted for an
        # account that ALREADY has an active, legitimate SSO session right
        # now is a genuinely suspicious pattern -- it suggests someone
        # other than the signed-in employee is attempting a password-based
        # entry on the same account. This connects two systems that
        # previously had no awareness of each other.
        active_sso = get_active_sessions(conn, email)
        suspicious_concurrent = bool(active_sso)

        row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (email,)).fetchone()

        if row:
            signals = json.loads(row["signals"])
            breach_list_for_new = None
        else:
            # No existing record -- this can happen if someone checks their
            # password or signs in via SSO before any CSV upload has run
            # for them. Previously this silently assumed email_breached:
            # False without ever checking; now it actually verifies, so
            # the result is correct regardless of which action happens
            # first for a given employee.
            breach_result = check_email_breaches(email)
            analytics_result = breach_analytics(email)
            signals = {
                "email_breached": bool(breach_result.get("breached")) if breach_result.get("status") != "error" else False,
                "dark_web_recent": bool(analytics_result.get("paste_exposure_count", 0) > 0) if analytics_result.get("status") != "error" else False,
            }
            breach_list_for_new = []
            for b in analytics_result.get("breach_details", []) or []:
                breach_list_for_new.append({
                    "name": b.get("breach", "Unnamed breach"),
                    "date": b.get("xposed_date", "unknown date"),
                    "added": b.get("added", ""),
                    "exposed_data": b.get("xposed_data", "unspecified"),
                    "records_exposed": b.get("xposed_records"),
                    "password_risk": b.get("password_risk", "unknown"),
                    "industry": b.get("industry", ""),
                    "description": b.get("details", ""),
                })

        priv = json.loads(row["privilege_detail"]) if row else compute_privilege_multiplier({}, 0)
        behavior_detail = json.loads(row["behavior_detail"]) if row else {
            "behavior_score": None, "flags": ["Login behavior not checked -- no login history provided"], "is_anomalous": None,
        }
        signals["password_reused"] = is_unsafe

        likelihood_pct = bayesian_likelihood(signals)
        final_score = min(100.0, round(likelihood_pct * priv.get("privilege_multiplier", 1.0), 1))
        severity = severity_label(final_score)

        finding_for_ai = {
            "employee_id": email, "signals": signals, "privilege": "regular",
            "final_risk_score": final_score, "severity": severity,
            "behavior_note": behavior_detail["flags"][0],
        }
        ai_result = get_recommendation(finding_for_ai, use_real_ai=False)
        recommendation_text = ai_result["recommendation"]

        if suspicious_concurrent:
            # Override, same pattern as the SSO-vs-SSO concurrent check --
            # this is treated as maximum-confidence suspicious activity,
            # not blended into the normal signal weighting.
            final_score = 100.0
            severity = "Critical"
            active_location = active_sso[0]["login_location"] or "an active session"
            recommendation_text = (
                f"URGENT: A password was just checked for this account while it already has an active "
                f"SSO session (signed in from {active_location}). This is consistent with someone other "
                f"than the account owner attempting access. Lock this account and contact the employee immediately."
            )

        conn.execute(
            """INSERT INTO findings (employee_id, risk_score, severity, signals,
               privilege_detail, behavior_detail, recommendation,
               data_source, password_checked, behavior_checked, access_checked,
               breach_detail_json, password_reset_required)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, 0)
               ON CONFLICT(employee_id) DO UPDATE SET
               risk_score=excluded.risk_score, severity=excluded.severity,
               signals=excluded.signals, recommendation=excluded.recommendation,
               password_checked=1, password_reset_required=0, updated_at=CURRENT_TIMESTAMP""",
            (
                email, final_score, severity, json.dumps(signals),
                json.dumps(priv), json.dumps(behavior_detail), recommendation_text,
                row["data_source"] if row else "real_upload",
                (row["behavior_checked"] if row else 0), (row["access_checked"] if row else 0),
                json.dumps(breach_list_for_new) if breach_list_for_new is not None else (row["breach_detail_json"] if row else "[]"),
            ),
        )
        conn.commit()

        termination_time = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        conn.execute(
            "UPDATE sessions SET logout_at = ? WHERE employee_id = ? AND logout_at IS NULL",
            (termination_time, email),
        )
        conn.commit()

    recompute_ml_flags()

    return {
        "status": "ok",
        "password_safe": not is_unsafe,
        "message": (
            "This password has been found in known data breaches. Please choose a different one."
            if is_unsafe else
            "Good news -- this password was not found in any known breach."
        ),
        "sessions_terminated": True,
        "note": "Any other active sessions for this account have been signed out as a precaution -- they'll need to sign in again.",
    }


@app.post("/api/upload-employees")
async def upload_employees(file: UploadFile = File(...)):
    """Accepts a CSV with a required 'email' column, and optional access-flag
    columns (admin_rights, financial_access, customer_data_access,
    source_code_access, hr_data_access, systems_accessible).

    Only email breach + dark-web/paste exposure can be checked automatically
    from an email alone -- password and login-behavior checks require the
    employee's own password entry / real login history, so those are marked
    NOT CHECKED rather than silently assumed safe.
    """
    # utf-8-sig strips a leading byte-order-mark if present, and behaves
    # identically to plain utf-8 if not -- PowerShell's "Out-File -Encoding
    # utf8" writes a BOM by default, which would otherwise silently corrupt
    # the first column name (turning "email" into an unmatched "\ufeffemail").
    raw = (await file.read()).decode("utf-8-sig")
    reader = csv_module.DictReader(StringIO(raw))

    if reader.fieldnames is None or "email" not in [f.strip().lower() for f in reader.fieldnames]:
        raise HTTPException(400, "CSV must have an 'email' column")

    rows = list(reader)
    # Exactly one row = someone adding a single new employee (visible in the
    # dashboard by default). More than one row = a bulk company upload,
    # treated as background comparison data -- still fully included in every
    # ML comparison, just not cluttering the default table view.
    source_label = "new_hire" if len(rows) == 1 else "real_upload"

    results = []
    for row in rows:
        row = {k.strip().lower(): v.strip() for k, v in row.items() if k}
        email = row.get("email")
        if not email:
            continue

        # --- Real, live checks: email breach + paste/dark-web exposure ---
        breach_result = check_email_breaches(email)
        analytics_result = breach_analytics(email)

        if breach_result.get("status") == "error" or analytics_result.get("status") == "error":
            results.append({
                "employee_id": email,
                "status": "error",
                "detail": breach_result.get("detail") or analytics_result.get("detail"),
            })
            continue

        signals = {
            "email_breached": bool(breach_result.get("breached")),
            "dark_web_recent": bool(analytics_result.get("paste_exposure_count", 0) > 0),
            # password_reused deliberately omitted: cannot be determined from
            # an email alone, and treating "unchecked" as "false" would be a
            # dangerous false negative in a security tool.
        }

        # Capture the actual per-breach detail the API already returns --
        # breach name, date, and what data types were exposed. Previously
        # only a plain true/false was kept; this is the same underlying
        # data XposedOrNot's own consumer-facing report shows, just for an
        # analyst investigating one employee instead of a self-lookup.
        breach_list = []
        for b in analytics_result.get("breach_details", []) or []:
            breach_list.append({
                "name": b.get("breach", "Unnamed breach"),
                "date": b.get("xposed_date", "unknown date"),
                "added": b.get("added", ""),
                "exposed_data": b.get("xposed_data", "unspecified"),
                "records_exposed": b.get("xposed_records"),
                "password_risk": b.get("password_risk", "unknown"),
                "industry": b.get("industry", ""),
                "description": b.get("details", ""),
            })

        password_strength = analytics_result.get("password_strength", {})
        exposed_data_types = analytics_result.get("exposed_data_types", {})

        # Optional additional dark-web signal source, requires its own separate
        # API key (INTELX_KEY). If not configured, this is skipped honestly --
        # never silently treated as "nothing found there either."
        dark_web_note = None
        intelx_key = get_intelx_key()
        if intelx_key:
            dw_result = search_dark_web(email, intelx_key, max_results=5)
            if dw_result.get("status") == "ok" and dw_result.get("found_on_dark_web"):
                signals["dark_web_recent"] = True
                dark_web_note = f"Dark web search: {dw_result['hit_count']} hit(s) found"
            elif dw_result.get("status") == "error":
                dark_web_note = f"Dark web search unavailable: {dw_result.get('detail')}"
        else:
            dark_web_note = "Dark web search not configured (INTELX_KEY not set) -- only paste-site exposure was checked"

        # --- Access rights: only if the CSV actually provided them ---
        access_provided = any(row.get(k) for k in
            ["admin_rights", "financial_access", "customer_data_access", "source_code_access", "hr_data_access"])
        access_flags = {
            k: row.get(k, "").lower() in ("1", "true", "yes")
            for k in ["admin_rights", "financial_access", "customer_data_access", "source_code_access", "hr_data_access"]
        }
        systems_accessible = int(row.get("systems_accessible", 0) or 0)
        privilege_result = compute_privilege_multiplier(access_flags, systems_accessible)

        likelihood_pct = bayesian_likelihood(signals)
        final_score = min(100.0, round(likelihood_pct * privilege_result["privilege_multiplier"], 1))
        severity = severity_label(final_score)

        behavior_detail = {
            "behavior_score": None,
            "flags": ["Login behavior not checked -- no login history provided"],
            "is_anomalous": None,
        }

        # Real recommendation, matched to actual severity -- not a hardcoded
        # string regardless of risk. Still honestly notes what wasn't checked.
        finding_for_ai = {
            "employee_id": email,
            "signals": signals,
            "privilege": "regular",
            "final_risk_score": final_score,
            "severity": severity,
            "behavior_note": "Login behavior not checked -- no login history provided",
        }
        ai_result = get_recommendation(finding_for_ai, use_real_ai=False)
        recommendation = (
            ai_result["recommendation"]
            + " Note: password status was not verified for this account -- "
            "ask the employee to reset it through the normal sign-in flow so it can be safely checked."
            + (f" {dark_web_note}." if dark_web_note else "")
        )

        with get_db() as conn:
            conn.execute(
                """INSERT INTO findings (employee_id, risk_score, severity, signals,
                   privilege_detail, behavior_detail, recommendation,
                   data_source, password_checked, behavior_checked, access_checked,
                   breach_detail_json, password_strength_json, exposed_data_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(employee_id) DO UPDATE SET
                   risk_score=excluded.risk_score, severity=excluded.severity,
                   signals=excluded.signals, privilege_detail=excluded.privilege_detail,
                   behavior_detail=excluded.behavior_detail, recommendation=excluded.recommendation,
                   data_source=excluded.data_source, password_checked=excluded.password_checked,
                   behavior_checked=excluded.behavior_checked, access_checked=excluded.access_checked,
                   breach_detail_json=excluded.breach_detail_json,
                   password_strength_json=excluded.password_strength_json,
                   exposed_data_json=excluded.exposed_data_json,
                   updated_at=CURRENT_TIMESTAMP""",
                (
                    email, final_score, severity, json.dumps(signals),
                    json.dumps(privilege_result), json.dumps(behavior_detail),
                    recommendation,
                    source_label, 0, 0, 1 if access_provided else 0,
                    json.dumps(breach_list), json.dumps(password_strength), json.dumps(exposed_data_types),
                ),
            )
            conn.commit()

        results.append({"employee_id": email, "status": "ok", "risk_score": final_score, "severity": severity})

    ml_result = recompute_ml_flags()

    # Rank each successfully-processed employee against the FULL current
    # population, sorted by risk -- this is the actual comparison-to-everyone
    # step, made visible in the response instead of just silently updating
    # the table.
    with get_db() as conn:
        all_sorted = conn.execute(
            "SELECT employee_id FROM findings ORDER BY risk_score DESC"
        ).fetchall()
    total = len(all_sorted)
    rank_by_id = {row["employee_id"]: i + 1 for i, row in enumerate(all_sorted)}

    for r in results:
        if r["status"] == "ok":
            r["rank"] = rank_by_id.get(r["employee_id"])
            r["total_employees"] = total

    return {"status": "ok", "processed": len(results), "results": results, "ml_cross_check": ml_result}


@app.get("/api/session-history/{employee_id}")
def session_history(employee_id: str):
    """Every recorded session for this employee -- login and logout times
    together -- not just their current status. This is what 'View
    Sessions' shows."""
    now_iso = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat()
    with get_db() as conn:
        rows = conn.execute(
            "SELECT login_at, logout_at, expires_at, login_location FROM sessions "
            "WHERE employee_id = ? ORDER BY login_at DESC LIMIT 50",
            (employee_id,),
        ).fetchall()
    history = []
    for r in rows:
        if r["logout_at"]:
            state = "signed_out"
        elif r["expires_at"] and r["expires_at"] < now_iso:
            state = "expired"
        else:
            state = "active"
        history.append({
            "login_at": r["login_at"],
            "logout_at": r["logout_at"],
            "login_location": r["login_location"],
            "state": state,
        })
    return {"employee_id": employee_id, "sessions": history}


def get_session_status(conn, employee_id):
    """The actual answer to 'when did they log in, are they still signed
    in, when did they log out' -- built from the server-side sessions
    table, not the employee's own browser cookie which the admin can't see."""
    latest = conn.execute(
        "SELECT login_at, expires_at, logout_at, login_location FROM sessions WHERE employee_id = ? ORDER BY login_at DESC LIMIT 1",
        (employee_id,),
    ).fetchone()
    if not latest:
        return {"state": "never_signed_in", "detail": "No SSO sign-in on record", "login_at": None, "active_sessions": []}

    # Every session currently active right now, not just the most recent --
    # this is what lets two simultaneous devices both show up, each with
    # their own login time, instead of only the latest one being visible.
    active_list = [dict(r) for r in get_active_sessions(conn, employee_id)]

    if latest["logout_at"]:
        return {"state": "signed_out", "detail": f"Signed out at {latest['logout_at']}", "login_at": latest["login_at"], "active_sessions": active_list}

    now_iso = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat()
    if latest["expires_at"] and latest["expires_at"] < now_iso:
        return {"state": "expired", "detail": f"Session expired (signed in {latest['login_at']}, no explicit logout)", "login_at": latest["login_at"], "active_sessions": active_list}

    if len(active_list) > 1:
        devices_desc = "; ".join(f"device signed in at {s['login_at']} from {s['login_location'] or 'unknown location'}" for s in active_list)
        return {"state": "active_multi", "detail": f"{len(active_list)} devices currently active \u2014 {devices_desc}", "login_at": latest["login_at"], "active_sessions": active_list}

    return {"state": "active", "detail": f"Currently signed in since {latest['login_at']} from {latest['login_location'] or 'unknown location'} (expires {latest['expires_at']})", "login_at": latest["login_at"], "active_sessions": active_list}


def get_active_sessions(conn, employee_id):
    """All sessions for this employee that are genuinely still active right
    now (not logged out, not expired) -- if there's more than one, that's
    the actual basis for concurrent-access detection."""
    now_iso = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat()
    return conn.execute(
        "SELECT id, login_at, login_location FROM sessions WHERE employee_id = ? AND logout_at IS NULL AND expires_at > ? ORDER BY login_at DESC",
        (employee_id, now_iso),
    ).fetchall()


_HERE = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
@app.get("/dashboard")
def serve_dashboard():
    """Serves dashboard.html as a real URL, not just a local file only
    reachable on the PC that has it -- this is what makes it accessible
    from a phone too, e.g. via 'adb reverse tcp:8001 tcp:8001'."""
    path = os.path.join(_HERE, "dashboard.html")
    if not os.path.exists(path):
        raise HTTPException(404, "dashboard.html not found in this folder")
    return FileResponse(path, media_type="text/html")


@app.get("/portal")
def serve_portal():
    """Serves employee_password_check.html the same way -- reachable from
    any device on the same tunnel/network, not just double-clicked locally."""
    path = os.path.join(_HERE, "employee_password_check.html")
    if not os.path.exists(path):
        raise HTTPException(404, "employee_password_check.html not found in this folder")
    return FileResponse(path, media_type="text/html")


@app.get("/my-status")
def serve_employee_dashboard():
    """The employee's own personal dashboard, shown after signing in --
    their own risk status only, never anyone else's."""
    path = os.path.join(_HERE, "employee_dashboard.html")
    if not os.path.exists(path):
        raise HTTPException(404, "employee_dashboard.html not found in this folder")
    return FileResponse(path, media_type="text/html")


@app.get("/api/login-locations")
def login_locations():
    """Real coordinates from actual recorded logins, for plotting on the
    map -- only points that have genuine lat/lon (not every login event
    has coordinates, e.g. ones that failed geolocation)."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT employee_id, login_location, lat, lon, login_hour, created_at FROM login_events "
            "WHERE lat IS NOT NULL AND lon IS NOT NULL ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/employees")
def list_employees():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM findings ORDER BY risk_score DESC"
        ).fetchall()
        session_map = {r["employee_id"]: get_session_status(conn, r["employee_id"]) for r in rows}
        # Most recent feedback per employee -- this is what lets "reviewed"
        # status genuinely survive navigation, since it's real backend data
        # rather than a one-off DOM change that resets on re-render.
        feedback_rows = conn.execute(
            "SELECT employee_id, verdict, MAX(created_at) as latest FROM feedback GROUP BY employee_id"
        ).fetchall()
        feedback_map = {r["employee_id"]: {"verdict": r["verdict"], "at": r["latest"]} for r in feedback_rows}
    return [
        {
            "employee_id": r["employee_id"],
            "risk_score": r["risk_score"],
            "severity": r["severity"],
            "signals": json.loads(r["signals"]),
            "privilege_detail": json.loads(r["privilege_detail"]),
            "behavior_detail": json.loads(r["behavior_detail"]),
            "recommendation": r["recommendation"],
            "data_source": r["data_source"],
            "password_checked": bool(r["password_checked"]),
            "behavior_checked": bool(r["behavior_checked"]),
            "access_checked": bool(r["access_checked"]),
            "ml_flagged": bool(r["ml_flagged"]),
            "ml_reason": r["ml_reason"],
            "ml_contributions": json.loads(r["ml_contributions_json"]) if r["ml_contributions_json"] else [],
            "breach_detail": json.loads(r["breach_detail_json"]) if r["breach_detail_json"] else [],
            "password_strength": json.loads(r["password_strength_json"]) if r["password_strength_json"] else {},
            "exposed_data_types": json.loads(r["exposed_data_json"]) if r["exposed_data_json"] else {},
            "session_status": session_map.get(r["employee_id"], {"state": "never_signed_in", "detail": "No SSO sign-in on record", "login_at": None}),
            "last_feedback": feedback_map.get(r["employee_id"]),
            "password_reset_required": bool(r["password_reset_required"]),
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


# In-memory pending OAuth states, for CSRF protection during the flow.
# Fine for this scope (a single-process demo server) -- a production
# deployment would use signed cookies instead.
_pending_oauth_states = set()


@app.get("/auth/login")
def sso_login():
    """Step 1: send the browser to Google's own consent screen. We never
    see or touch the employee's actual Google password -- that happens
    entirely on Google's side."""
    if not google_sso.is_configured():
        return HTMLResponse(
            "<h2>Google SSO not configured</h2>"
            "<p>Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET first. "
            "See google_sso.py for setup steps.</p>", status_code=503,
        )
    state = secrets.token_urlsafe(24)
    _pending_oauth_states.add(state)
    return RedirectResponse(google_sso.build_auth_url(state))


@app.post("/auth/update-location")
async def update_location(request: Request, lat: float = Form(...), lon: float = Form(...)):
    """Called from the confirmation page's JS after the browser grants real
    GPS/WiFi location permission -- upgrades the just-recorded login event
    from IP-based guessing to actual precise coordinates, AND re-scores the
    behavior check against this corrected location so the finding doesn't
    silently go stale relative to the more accurate data now on record."""
    email = request.cookies.get("session_email")
    if not email:
        raise HTTPException(401, "Not signed in")

    geo = google_sso.reverse_geocode(lat, lon)
    real_location = geo["location"]

    with get_db() as conn:
        latest = conn.execute(
            "SELECT id, login_hour FROM login_events WHERE employee_id = ? ORDER BY created_at DESC LIMIT 1",
            (email,),
        ).fetchone()
        if not latest:
            return {"status": "ok", "location": real_location, "rescored": False}

        conn.execute(
            "UPDATE login_events SET login_location = ?, lat = ?, lon = ? WHERE id = ?",
            (real_location, geo["lat"], geo["lon"], latest["id"]),
        )
        conn.commit()

        # Re-score against everything EXCEPT this just-corrected row, using
        # the real location instead of the stale IP-based guess.
        past = conn.execute(
            "SELECT login_hour, login_location FROM login_events WHERE employee_id = ? AND id != ? ORDER BY created_at",
            (email, latest["id"]),
        ).fetchall()

        behavior_result = None
        if len(past) >= 3:
            baseline = build_baseline([{"hour": r["login_hour"], "location": r["login_location"]} for r in past])
            behavior_result = score_login(baseline, {"hour": latest["login_hour"], "location": real_location})

            row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (email,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE findings SET behavior_detail = ?, behavior_checked = 1, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                    (json.dumps(behavior_result), email),
                )
                conn.commit()
                recompute_ml_flags()

    return {"status": "ok", "location": real_location, "rescored": behavior_result is not None, "behavior_result": behavior_result}


@app.get("/auth/callback")
def sso_callback(request: Request, code: str = None, state: str = None, error: str = None):
    """Step 2: Google redirects back here with a one-time code. We trade it
    for the user's verified email, record this as a REAL login event, and
    -- once enough real logins exist for this person -- score it against
    their own accumulated history, exactly like the manual test tool does,
    just with genuine data instead of typed-in guesses."""
    if error:
        return HTMLResponse(f"<h2>Sign-in cancelled</h2><p>{error}</p>", status_code=400)
    if not state or state not in _pending_oauth_states:
        return HTMLResponse("<h2>Invalid or expired sign-in attempt</h2><p>Please try again.</p>", status_code=400)
    _pending_oauth_states.discard(state)

    token_result = google_sso.exchange_code_for_token(code)
    if token_result.get("status") == "error":
        return HTMLResponse(f"<h2>Sign-in failed</h2><p>{token_result['detail']}</p>", status_code=502)

    user_info = google_sso.get_user_info(token_result["access_token"])
    if user_info.get("status") == "error":
        return HTMLResponse(f"<h2>Could not verify identity</h2><p>{user_info['detail']}</p>", status_code=502)

    email = user_info.get("email")
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    client_ip = request.client.host if request.client else "unknown"
    geo = google_sso.geolocate_ip(client_ip)
    location, geo_lat, geo_lon = geo["location"], geo["lat"], geo["lon"]
    login_at_ist = now.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
    expires_at = (now.replace(tzinfo=None) + timedelta(hours=5)).isoformat()

    concurrent_warning = None
    with get_db() as conn:
        # Check for concurrent access BEFORE this new session exists -- any
        # session for this person still active right now, from a genuinely
        # different location, is a strong real-time compromise signal: it
        # means two people (or the same person impossibly fast) are signed
        # in from different places at once. This is a simplified version of
        # what real enterprise tools call "impossible travel" detection --
        # it doesn't calculate real-world distance/speed feasibility, just
        # flags any concurrent access from a different location, which is
        # still a genuine, useful signal for exactly the scenario it's
        # meant to catch.
        active_elsewhere = conn.execute(
            "SELECT login_location FROM sessions WHERE employee_id = ? AND logout_at IS NULL AND expires_at > ? AND login_location != ? AND login_location IS NOT NULL",
            (email, login_at_ist, location),
        ).fetchall()
        if active_elsewhere:
            other_locations = ", ".join(sorted(set(r["login_location"] for r in active_elsewhere)))
            concurrent_warning = f"CONCURRENT ACCESS DETECTED: an active session already exists from '{other_locations}' while this new one is from '{location}' \u2014 possible account compromise."

        conn.execute(
            "INSERT INTO sessions (employee_id, login_at, expires_at, login_location) VALUES (?, ?, ?, ?)",
            (email, login_at_ist, expires_at, location),
        )
        conn.commit()

        # Build the baseline from this employee's REAL past logins (before
        # recording this one), so today's login is judged against genuine
        # history, not itself.
        past = conn.execute(
            "SELECT login_hour, login_location FROM login_events WHERE employee_id = ? ORDER BY created_at DESC LIMIT 20",
            (email,),
        ).fetchall()

        conn.execute(
            "INSERT INTO login_events (employee_id, login_hour, login_location, ip_address, lat, lon) VALUES (?, ?, ?, ?, ?, ?)",
            (email, now.hour, location, client_ip, geo_lat, geo_lon),
        )
        conn.commit()

        if len(past) < 3:
            behavior_summary = ""
        else:
            baseline = build_baseline([{"hour": r["login_hour"], "location": r["login_location"]} for r in past])
            result = score_login(baseline, {"hour": now.hour, "location": location})
            behavior_summary = f"Score {result['behavior_score']} \u2014 {'FLAGGED' if result['is_anomalous'] else 'normal'}. {'; '.join(result['flags'])}"

            row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (email,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE findings SET behavior_detail = ?, behavior_checked = 1, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                    (json.dumps(result), email),
                )
                conn.commit()
                recompute_ml_flags()

        # Concurrent access from a different location overrides everything
        # else -- treated as maximum-confidence compromise, regardless of
        # what the normal Bayesian/behavior math would otherwise say. This
        # is deliberately NOT blended into the existing signal weighting;
        # keeping it as a hard override keeps its meaning unambiguous both
        # here and when explaining it in a viva.
        if concurrent_warning:
            behavior_summary = f"\u26a0 {concurrent_warning} {behavior_summary}"
            row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (email,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE findings SET risk_score = 100.0, severity = 'Critical', "
                    "recommendation = ?, updated_at = CURRENT_TIMESTAMP WHERE employee_id = ?",
                    (f"URGENT: {concurrent_warning} Lock this account and force a password reset immediately.", email),
                )
                conn.commit()
                recompute_ml_flags()

    response = HTMLResponse(f"""
        <div style="font-family: -apple-system, sans-serif; max-width: 480px; margin: 60px auto; text-align: center;">
            <h2>Signed in as {email}</h2>
            <p>Login recorded: {now.strftime('%H:%M IST')} from <span id="loc">{location}</span></p>
            <p id="behaviorSummary" style="color: #555; display:{'none' if not behavior_summary else 'block'};">{behavior_summary}</p>
            <p style="margin-top: 24px;"><a href="/my-status" style="background:#0c447c; color:white; padding:9px 18px; border-radius:6px; text-decoration:none; font-size:13px; font-weight:500;">View my security status</a></p>
            <p style="margin-top: 16px;">
                <a href="/auth/login">Sign in again to build more history</a> &middot;
                <a href="/auth/logout">Sign out</a>
            </p>
            <p id="geoStatus" style="font-size: 12px; color: #999; margin-top: 16px;"></p>
        </div>
        <script>
        // Upgrade from IP-based location to real GPS/WiFi positioning, if
        // the browser grants permission -- IP geolocation is only regionally
        // accurate (nearby cities on the same ISP can be indistinguishable),
        // real device positioning is far more precise.
        const geoStatus = document.getElementById("geoStatus");
        if (navigator.geolocation) {{
            geoStatus.textContent = "Requesting precise location...";
            navigator.geolocation.getCurrentPosition(async (pos) => {{
                geoStatus.textContent = "Got device location, looking up address...";
                try {{
                    const formData = new FormData();
                    formData.append("lat", pos.coords.latitude);
                    formData.append("lon", pos.coords.longitude);
                    const res = await fetch("/auth/update-location", {{ method: "POST", body: formData, credentials: "include" }});
                    const data = await res.json();
                    if (data.status === "ok") {{
                        document.getElementById("loc").textContent = data.location;
                        if (data.rescored && data.behavior_result) {{
                            const r = data.behavior_result;
                            const summaryEl = document.getElementById("behaviorSummary");
                            summaryEl.style.display = "block";
                            summaryEl.textContent =
                                "Score " + r.behavior_score + " \u2014 " + (r.is_anomalous ? "FLAGGED" : "normal") + ". " + r.flags.join("; ") + " (updated with your real location)";
                        }}
                        geoStatus.textContent = "";
                    }} else {{
                        geoStatus.textContent = "Location update failed: " + JSON.stringify(data);
                    }}
                }} catch (e) {{ geoStatus.textContent = "Location update request failed: " + e; }}
            }}, (err) => {{
                // err.code: 1 = permission denied, 2 = position unavailable, 3 = timeout
                geoStatus.textContent = "Precise location not available (code " + err.code + ": " + err.message + ") -- using IP-based estimate instead.";
            }}, {{ timeout: 20000 }});
        }} else {{
            geoStatus.textContent = "This browser does not support precise geolocation -- using IP-based estimate instead.";
        }}
        </script>
    """)
    # A simple demo-scope session cookie -- good enough to make "signed in"
    # a real, persistent state rather than a one-off event with nothing to
    # log out of. A production system would sign/encrypt this properly.
    # Session automatically expires after 5 hours of inactivity, then
    # requires signing in again -- a reasonable middle ground between
    # security (sessions shouldn't last forever) and not being mistaken
    # for a logout that never happened.
    response.set_cookie("session_email", email, max_age=5 * 3600, httponly=True)
    return response


@app.get("/auth/logout")
def sso_logout(request: Request):
    """Clears the session cookie -- the thing that was missing before --
    and records the explicit logout server-side, so the admin dashboard
    can distinguish 'still active' from 'this person deliberately signed
    out' rather than just watching the timer run out."""
    email = request.cookies.get("session_email")
    if email:
        logout_at_ist = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        with get_db() as conn:
            conn.execute(
                "UPDATE sessions SET logout_at = ? WHERE employee_id = ? AND logout_at IS NULL",
                (logout_at_ist, email),
            )
            conn.commit()

    response = HTMLResponse("""
        <div style="font-family: -apple-system, sans-serif; max-width: 480px; margin: 60px auto; text-align: center;">
            <h2>Signed out</h2>
            <p><a href="/auth/login">Sign in again</a></p>
        </div>
    """)
    response.delete_cookie("session_email")
    return response


@app.get("/auth/session")
def sso_session(request: Request):
    """Lets the employee portal page check whether someone's currently
    signed in, since JS can't read an httponly cookie directly."""
    email = request.cookies.get("session_email")
    return {"signed_in": bool(email), "email": email}


@app.get("/api/my-status")
def my_status(request: Request):
    """The employee-facing counterpart to /api/employees -- returns only
    the signed-in person's own record, never anyone else's. This is what
    powers the personal dashboard shown after signing in."""
    email = request.cookies.get("session_email")
    if not email:
        raise HTTPException(401, "Not signed in")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM findings WHERE employee_id = ?", (email,)).fetchone()
        session_status = get_session_status(conn, email)

        recent_logins = conn.execute(
            "SELECT login_hour, login_location, created_at FROM login_events WHERE employee_id = ? ORDER BY created_at DESC LIMIT 5",
            (email,),
        ).fetchall()

    if not row:
        return {
            "found": False,
            "email": email,
            "session_status": session_status,
            "message": "No assessment on record yet for this account.",
        }

    return {
        "found": True,
        "email": email,
        "risk_score": row["risk_score"],
        "severity": row["severity"],
        "signals": json.loads(row["signals"]),
        "recommendation": row["recommendation"],
        "password_checked": bool(row["password_checked"]),
        "behavior_checked": bool(row["behavior_checked"]),
        "breach_detail": json.loads(row["breach_detail_json"]) if row["breach_detail_json"] else [],
        "password_strength": json.loads(row["password_strength_json"]) if row["password_strength_json"] else {},
        "session_status": session_status,
        "recent_logins": [dict(r) for r in recent_logins],
        "password_reset_required": bool(row["password_reset_required"]),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
