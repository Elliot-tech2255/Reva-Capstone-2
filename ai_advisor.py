"""
Credential Exposure Monitor -- AI Advisor (Step 4: What You See)
The last piece of the pipeline. By the time this runs, the score is
ALREADY decided by math (scoring_engine.py) and ML (anomaly_detection_check.py).
This module's only job: turn a finished decision into one clear sentence.

It never re-evaluates whether someone is dangerous -- it only explains a
verdict that already exists.

SETUP (do this once you have your own key):
    pip install google-genai
    export GEMINI_API_KEY="your-key-here"

Until you add a real key, this runs in FALLBACK MODE -- a rule-based
template stands in for the AI, so the rest of the pipeline can still be
tested end-to-end without waiting on API access.
"""

import json
import os


def _build_prompt(finding: dict) -> str:
    """Turn a scored finding into a clear instruction for the model.
    Kept short and structured on purpose -- the model's only job is to
    phrase this, not decide anything new."""
    return f"""You are writing a ONE-SENTENCE security instruction for a busy IT
security analyst. Do not add caveats, do not question the finding, do not
add extra commentary -- the risk decision is already final. Just tell the
analyst exactly what action to take.

Employee: {finding['employee_id']}
Risk score: {finding['final_risk_score']}/100 ({finding['severity']})
Signals found: {", ".join(k for k, v in finding['signals'].items() if v) or "none beyond behavior"}
Account privilege: {finding['privilege']}
Behavior check: {finding.get('behavior_note', 'not checked')}

Write one short, direct sentence telling the analyst what to do right now."""


def _fallback_recommendation(finding: dict) -> str:
    """Rule-based stand-in used until a real Gemini key is configured.
    Deliberately simple and template-based -- this is NOT the real AI
    advisor, just a placeholder so the pipeline is testable end-to-end."""
    score = finding["final_risk_score"]
    signals = finding["signals"]

    if score >= 80:
        action = "Reset this password immediately and enable two-factor authentication."
    elif score >= 50:
        action = "Reset this password within 24 hours and review recent account activity."
    elif score >= 20:
        action = "Recommend a password change at the employee's next login."
    else:
        action = "No action needed -- continue routine monitoring."

    if signals.get("password_reused"):
        action += " This password is confirmed unsafe, so treat this as urgent."
    return action


def get_recommendation(finding: dict, use_real_ai: bool = False) -> dict:
    """Main entry point. Set use_real_ai=True once GEMINI_API_KEY is set."""
    if use_real_ai:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "detail": "GEMINI_API_KEY not set. Get one at aistudio.google.com/apikey",
                "recommendation": None,
            }
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = _build_prompt(finding)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            return {"status": "ok", "mode": "real_ai", "recommendation": response.text.strip()}
        except Exception as exc:
            return {"status": "error", "detail": str(exc), "recommendation": None}

    # Fallback mode -- no API key needed, lets you test the whole pipeline today
    return {
        "status": "ok",
        "mode": "fallback_template",
        "recommendation": _fallback_recommendation(finding),
    }


if __name__ == "__main__":
    # Priya's finding, exactly as scoring_engine.py would produce it
    priya_finding = {
        "employee_id": "priya@nexoraretail.com",
        "signals": {"email_breached": True, "password_reused": True, "dark_web_recent": True},
        "privilege": "admin",
        "final_risk_score": 100.0,
        "severity": "Critical",
        "behavior_note": "Login from an unfamiliar location at an unusual hour",
    }

    print("=== AI Advisor output (fallback mode -- no API key needed) ===")
    result = get_recommendation(priya_finding, use_real_ai=False)
    print(json.dumps(result, indent=2))

    print("\n=== Same finding, attempting real AI (will show setup instructions if no key set) ===")
    result_real = get_recommendation(priya_finding, use_real_ai=True)
    print(json.dumps(result_real, indent=2))
