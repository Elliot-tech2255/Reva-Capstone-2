"""
Credential Exposure Monitor -- Google SSO (real OAuth2/OIDC)
This is the SAME protocol real enterprise SSO (Okta, Azure AD) runs on
underneath -- just using Google as the identity provider instead of a
corporate one, since that's something anyone can register and test for
free without needing an actual company tenant.

Written by hand (not hidden behind an OAuth library) on purpose -- this
project's whole point is explainability, so every step of the exchange
should be readable, not abstracted away.

SETUP (one-time, free):
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create a project (or use an existing one)
    3. Configure the OAuth consent screen (External, Testing mode is fine)
    4. Create Credentials -> OAuth client ID -> Web application
    5. Add authorized redirect URI: http://localhost:8000/auth/callback
    6. Copy the Client ID and Client Secret, then:
         $env:GOOGLE_CLIENT_ID = "your-client-id"
         $env:GOOGLE_CLIENT_SECRET = "your-client-secret"
"""

import os
import json
import urllib.request
import urllib.parse
import urllib.error

def _load_credentials():
    """Environment variables first; falls back to a plain local file if
    those aren't visible to this process for any reason (proved unreliable
    in testing -- this file-based path has no dependency on shell session
    state, parent/child process inheritance, or which terminal window a
    command happened to run in)."""
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    cred_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "google_credentials.txt")
    if os.path.exists(cred_file):
        values = {}
        with open(cred_file, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
        return values.get("GOOGLE_CLIENT_ID"), values.get("GOOGLE_CLIENT_SECRET")

    return None, None


GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET = _load_credentials()
REDIRECT_URI = os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8001/auth/callback")

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
IP_GEOLOCATION_URL = "http://ip-api.com/json/{ip}"  # free tier, no key needed


def is_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def build_auth_url(state: str) -> str:
    """Step 1 of OAuth2: build the URL that sends the browser to Google's
    own consent screen. We never see the user's Google password -- Google
    handles that entirely on their side."""
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,  # CSRF protection -- verified again in the callback
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    """Step 2: trade the one-time authorization code Google gave us for an
    actual access token. This call happens server-to-server, never through
    the browser."""
    data = urllib.parse.urlencode({
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=data, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return {"status": "ok", **json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        return {"status": "error", "detail": f"http_{e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def get_user_info(access_token: str) -> dict:
    """Step 3: use the access token to ask Google who actually just signed
    in -- their verified email is what identifies which employee this is."""
    req = urllib.request.Request(USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return {"status": "ok", **json.loads(resp.read())}
    except urllib.error.HTTPError as e:
        return {"status": "error", "detail": f"http_{e.code}: {e.reason}"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def reverse_geocode(lat: float, lon: float) -> dict:
    """Turns real browser-reported coordinates (GPS/WiFi positioning, much
    more precise than IP-based lookup) into a city/country string, plus
    the coordinates themselves for mapping. Uses OpenStreetMap's Nominatim
    service -- free, no API key needed."""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    req = urllib.request.Request(url, headers={"User-Agent": "capstone-credential-monitor/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read())
            addr = data.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county") or "Unknown city"
            country = addr.get("country", "Unknown country")
            return {"location": f"{city}, {country}", "lat": lat, "lon": lon}
    except Exception:
        return {"location": "Unknown location", "lat": lat, "lon": lon}


def geolocate_ip(ip: str) -> dict:
    """Turns an IP into a city/country string plus coordinates, for the
    behavior check's location signal and for mapping. When the connecting
    IP is local (as it always is when testing on your own machine via
    localhost), falls back to looking up the server machine's own
    public-facing IP instead -- for local testing this is genuinely more
    useful, since it reflects where you actually are rather than a
    meaningless '127.0.0.1'. In a real deployment reached over the
    internet, request.client.host would already be a real public IP and
    this fallback simply wouldn't trigger."""
    is_local = ip in ("127.0.0.1", "::1", "localhost") or ip.startswith("192.168.") or ip.startswith("10.")
    lookup_url = "http://ip-api.com/json/" if is_local else IP_GEOLOCATION_URL.format(ip=ip)
    try:
        with urllib.request.urlopen(lookup_url, timeout=5) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "success":
                return {
                    "location": f"{data.get('city', 'Unknown city')}, {data.get('country', 'Unknown country')}",
                    "lat": data.get("lat"), "lon": data.get("lon"),
                }
    except Exception:
        pass
    return {"location": "Unknown location", "lat": None, "lon": None}
