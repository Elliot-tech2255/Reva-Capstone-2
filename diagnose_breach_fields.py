"""
One-time diagnostic: prints the RAW breach_details structure for a real
email, so we can see the actual field names instead of guessing again.
Run this directly, not through the dashboard.
"""
import json
from credential_exposure_check import breach_analytics

email = "lakshmiraj.gr@gmail.com"
result = breach_analytics(email)

print("=== Full raw response ===")
print(json.dumps(result, indent=2))

print()
print("=== Just the breach_details items, with their exact keys ===")
for b in result.get("breach_details", []):
    print("Keys found:", list(b.keys()))
    print(json.dumps(b, indent=2))
