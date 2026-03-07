"""OpenAI OAuth (PKCE) helpers for wizard login and token refresh."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from whai.constants import (
    DEFAULT_OPENAI_OAUTH_AUTHORIZE_URL,
    DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    DEFAULT_OPENAI_OAUTH_CALLBACK_PORT,
    DEFAULT_OPENAI_OAUTH_SCOPES,
    DEFAULT_OPENAI_OAUTH_TOKEN_URL,
)


def _base64url_no_padding(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def create_pkce_verifier() -> str:
    """Create an RFC 7636-compliant PKCE code verifier."""
    # 64 url-safe chars, inside the 43-128 char required range.
    return _base64url_no_padding(secrets.token_bytes(48))


def create_pkce_challenge(verifier: str) -> str:
    """Create S256 PKCE challenge from verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _base64url_no_padding(digest)


def create_oauth_state() -> str:
    """Create an opaque state value for CSRF protection."""
    return _base64url_no_padding(secrets.token_bytes(24))


def get_default_redirect_uri() -> str:
    """Return the default localhost callback URI."""
    return f"http://{DEFAULT_OPENAI_OAUTH_CALLBACK_HOST}:{DEFAULT_OPENAI_OAUTH_CALLBACK_PORT}/auth/callback"


def build_openai_authorize_url(
    oauth_client_id: str,
    code_challenge: str,
    state: str,
    redirect_uri: Optional[str] = None,
    scopes: str = DEFAULT_OPENAI_OAUTH_SCOPES,
) -> str:
    """Build the OpenAI OAuth authorization URL."""
    if not oauth_client_id or not oauth_client_id.strip():
        raise ValueError("oauth_client_id is required")

    redirect = redirect_uri or get_default_redirect_uri()
    params = {
        "response_type": "code",
        "client_id": oauth_client_id,
        "redirect_uri": redirect,
        "scope": scopes,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return f"{DEFAULT_OPENAI_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def parse_code_from_user_input(pasted: str, expected_state: str) -> str:
    """Parse authorization code from a pasted redirect URL or raw code."""
    text = (pasted or "").strip()
    if not text:
        raise ValueError("No authorization code was provided")

    # If user pasted a URL, extract query params.
    if text.startswith("http://") or text.startswith("https://"):
        parsed = urllib.parse.urlparse(text)
        query = urllib.parse.parse_qs(parsed.query)
        if "error" in query:
            error = query.get("error", ["unknown_error"])[0]
            raise ValueError(f"OAuth authorization failed: {error}")

        state = query.get("state", [""])[0]
        if expected_state and state and state != expected_state:
            raise ValueError("OAuth state mismatch; please retry login")

        code = query.get("code", [""])[0]
        if not code:
            raise ValueError("No 'code' found in pasted callback URL")
        return code

    # Assume user pasted the raw authorization code directly.
    return text


def _post_form(url: str, form_data: Dict[str, str]) -> Dict[str, Any]:
    body = urllib.parse.urlencode(form_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    """Decode JWT payload without verification to extract non-sensitive claims."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def extract_account_id(access_token: str) -> Optional[str]:
    """Extract account/user id from access token claims when available."""
    payload = _decode_jwt_payload(access_token)
    for key in ("account_id", "acct", "sub"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def exchange_code_for_tokens(
    oauth_client_id: str,
    authorization_code: str,
    code_verifier: str,
    redirect_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Exchange authorization code for access/refresh tokens."""
    redirect = redirect_uri or get_default_redirect_uri()
    payload = {
        "grant_type": "authorization_code",
        "client_id": oauth_client_id,
        "code": authorization_code,
        "code_verifier": code_verifier,
        "redirect_uri": redirect,
    }
    token_response = _post_form(DEFAULT_OPENAI_OAUTH_TOKEN_URL, payload)
    return _normalize_token_response(token_response)


def refresh_openai_tokens(
    oauth_client_id: str,
    refresh_token: str,
) -> Dict[str, Any]:
    """Refresh OpenAI OAuth tokens."""
    payload = {
        "grant_type": "refresh_token",
        "client_id": oauth_client_id,
        "refresh_token": refresh_token,
    }
    token_response = _post_form(DEFAULT_OPENAI_OAUTH_TOKEN_URL, payload)
    normalized = _normalize_token_response(token_response)
    if not normalized.get("refresh_token"):
        normalized["refresh_token"] = refresh_token
    return normalized


def _normalize_token_response(token_response: Dict[str, Any]) -> Dict[str, Any]:
    access_token = token_response.get("access_token")
    refresh_token = token_response.get("refresh_token")
    expires_in = token_response.get("expires_in")

    if not access_token:
        raise RuntimeError("OAuth token exchange did not return an access_token")

    try:
        expires_seconds = int(expires_in) if expires_in is not None else 3600
    except Exception:
        expires_seconds = 3600

    now = int(time.time())
    expires_at = now + max(60, expires_seconds)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "token_type": token_response.get("token_type", "Bearer"),
        "account_id": extract_account_id(access_token),
    }
