from __future__ import annotations

from base64 import urlsafe_b64encode
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid
import webbrowser

import httpx
import jwt
from appdirs import user_config_dir
import portalocker


AUTHORIZATION_ENDPOINT = "https://auth.openai.com/api/accounts/authorize"
TOKEN_ENDPOINT = "https://auth.openai.com/api/accounts/oauth/token"
REVOCATION_ENDPOINT = "https://auth.openai.com/api/accounts/oauth/revoke"
JWKS_URL = "https://auth.openai.com/.well-known/jwks.json"
ISSUER = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
PLAN_SCOPE = "chatgpt.tokens.use.direct"
OAUTH_SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"


class ChatGPTOAuthError(RuntimeError):
    """Raised when ChatGPT OAuth credentials cannot be used."""


class _CallbackServer(HTTPServer):
    callback: dict[str, str] | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404)
            return
        query = parse_qs(parsed.query)
        cast(_CallbackServer, self.server).callback = {key: values[0] for key, values in query.items() if values}
        body = b"ChatGPT authorization received. You can close this tab and return to CrewAI."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Authorization codes and OAuth errors arrive in the callback URL.
        return


class ChatGPTOAuth:
    """Local Sign in with ChatGPT session for eligible Responses API requests."""

    def __init__(self, config_dir: str | Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else Path(user_config_dir("crewai", "CrewAI"))
        self.profile_path = self.config_dir / "chatgpt-auth.json"
        self.host_id_path = self.config_dir / "chatgpt-host-id"
        self.lock_path = self.config_dir / "chatgpt-auth.lock"

    def sign_in(self, app_name: str, *, timeout: float = 300.0) -> dict[str, Any]:
        """Run the local OAuth authorization-code flow and save its credentials."""
        name = app_name.strip()
        if not name:
            raise ValueError("app_name must not be empty")

        profile = self._read_profile()
        issued_client_id = str(profile.get("client_id", "")) if profile else ""
        registration_client_id = issued_client_id or "dynamic_agent_client"
        host_id = self._get_or_create_host_id()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode("ascii").rstrip("=")

        server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
        server.timeout = 1.0
        redirect_uri = f"http://127.0.0.1:{server.server_port}/auth/callback"
        params: dict[str, str] = {
            "client_id": registration_client_id,
            "ext_agent_host_id": host_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": OAUTH_SCOPE,
            "resource": RESOURCE,
            "state": state,
            "nonce": nonce,
            "code_challenge_method": "S256",
            "code_challenge": challenge,
        }
        if issued_client_id:
            if profile and profile.get("id_token"):
                params["id_token_hint"] = str(profile["id_token"])
            if profile and profile.get("email"):
                params["login_hint"] = str(profile["email"])
        else:
            params["agent_name_hint"] = name

        authorization_url = f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}"
        if not webbrowser.open(authorization_url):
            sys.stdout.write("Open this URL to continue ChatGPT authorization:\n")
            sys.stdout.write(f"{authorization_url}\n")
            sys.stdout.flush()

        deadline = time.monotonic() + timeout
        try:
            while server.callback is None and time.monotonic() < deadline:
                server.handle_request()
        finally:
            server.server_close()

        callback = server.callback
        if callback is None:
            raise ChatGPTOAuthError("ChatGPT authorization timed out. Run crewai auth chatgpt login again.")
        if not secrets.compare_digest(callback.get("state", ""), state):
            raise ChatGPTOAuthError("The ChatGPT OAuth callback state did not match.")
        if callback.get("error"):
            if callback["error"] == "access_denied":
                raise ChatGPTOAuthError("ChatGPT plan usage was not authorized. No credentials were saved.")
            raise ChatGPTOAuthError(f"ChatGPT authorization failed with OAuth error: {callback['error']}")

        code = callback.get("code")
        if not code:
            raise ChatGPTOAuthError("The ChatGPT OAuth callback did not include a code.")

        callback_client_id = callback.get("client_id", "")
        if issued_client_id:
            if callback_client_id and callback_client_id != issued_client_id:
                raise ChatGPTOAuthError("The callback client ID did not match the saved registration.")
            client_id = issued_client_id
        else:
            client_id = callback_client_id
            if not client_id or client_id == "dynamic_agent_client":
                raise ChatGPTOAuthError("OpenAI did not return an issued client ID; registration is incomplete.")

        try:
            response = httpx.post(
                TOKEN_ENDPOINT,
                data={
                    "grant_type": "authorization_code",
                    "client_id": client_id,
                    "code": code,
                    "code_verifier": verifier,
                    "redirect_uri": redirect_uri,
                    "resource": RESOURCE,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            token_data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ChatGPTOAuthError("ChatGPT authorization code exchange failed. Start a fresh sign-in.") from exc

        scopes = set(str(token_data.get("scope", "")).split())
        if PLAN_SCOPE not in scopes:
            raise ChatGPTOAuthError("ChatGPT plan usage permission was not granted. No credentials were saved.")

        id_token = token_data.get("id_token")
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        if not all(isinstance(value, str) and value for value in (id_token, access_token, refresh_token)):
            raise ChatGPTOAuthError("OpenAI returned an incomplete token response. No credentials were saved.")

        identity = self._validate_id_token(id_token, client_id, nonce)
        if profile and profile.get("subject") and profile["subject"] != identity["sub"]:
            raise ChatGPTOAuthError("The selected ChatGPT account changed during reauthorization.")

        saved_profile: dict[str, Any] = {
            "client_id": client_id,
            "ext_agent_host_id": host_id,
            "issuer": ISSUER,
            "subject": identity["sub"],
            "email": identity.get("email"),
            "id_token": id_token,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": token_data.get("token_type", "Bearer"),
            "scopes": sorted(scopes),
            "expires_at": time.time() + self._positive_int(token_data.get("expires_in"), default=3600),
        }
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=30):
            self._write_profile(saved_profile)
        return self.status() or {}

    def get_access_token(self) -> str:
        """Return a current bearer token, refreshing it when close to expiry."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._set_directory_permissions()
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=30):
            profile = self._read_profile()
            if profile is None:
                raise ChatGPTOAuthError("No ChatGPT plan connection is saved. Run crewai auth chatgpt login first.")
            self._ensure_plan_scope(profile)

            access_token = profile.get("access_token")
            if isinstance(access_token, str) and access_token and float(profile.get("expires_at", 0)) > time.time() + 60:
                return access_token

            refresh_token = profile.get("refresh_token")
            client_id = profile.get("client_id")
            if not isinstance(refresh_token, str) or not refresh_token or not isinstance(client_id, str) or not client_id:
                raise ChatGPTOAuthError("The ChatGPT session needs authorization again. Run crewai auth chatgpt login.")

            try:
                response = httpx.post(
                    TOKEN_ENDPOINT,
                    data={
                        "grant_type": "refresh_token",
                        "client_id": client_id,
                        "refresh_token": refresh_token,
                        "resource": RESOURCE,
                    },
                    timeout=30.0,
                )
                if response.status_code == 400:
                    try:
                        error_data = response.json()
                    except ValueError:
                        error_data = {}
                    if error_data.get("error") in {
                        "invalid_grant", "invalid_refresh_token", "refresh_token_expired",
                        "refresh_token_invalidated", "refresh_token_reused", "token_expired",
                    }:
                        profile["access_token"] = ""
                        profile["refresh_token"] = ""
                        profile["expires_at"] = 0
                        self._write_profile(profile)
                        raise ChatGPTOAuthError("The ChatGPT refresh token is no longer valid. Run crewai auth chatgpt login.")
                response.raise_for_status()
                refreshed = response.json()
            except ChatGPTOAuthError:
                raise
            except (httpx.HTTPError, ValueError) as exc:
                raise ChatGPTOAuthError("Could not refresh the ChatGPT session. Retry later or sign in again.") from exc

            new_access_token = refreshed.get("access_token")
            if not isinstance(new_access_token, str) or not new_access_token:
                raise ChatGPTOAuthError("OpenAI returned no access token during refresh. Sign in again.")
            scopes = set(str(refreshed.get("scope", "")).split()) or set(profile.get("scopes", []))
            if PLAN_SCOPE not in scopes:
                profile["access_token"] = ""
                profile["refresh_token"] = ""
                profile["expires_at"] = 0
                self._write_profile(profile)
                raise ChatGPTOAuthError("The refreshed ChatGPT session no longer has plan usage permission.")

            profile["access_token"] = new_access_token
            profile["refresh_token"] = refreshed.get("refresh_token") or refresh_token
            profile["scopes"] = sorted(scopes)
            profile["expires_at"] = time.time() + self._positive_int(refreshed.get("expires_in"), default=3600)
            self._write_profile(profile)
            return new_access_token

    def status(self) -> dict[str, Any] | None:
        """Return account status without exposing OAuth tokens."""
        profile = self._read_profile()
        if profile is None:
            return None
        self._ensure_plan_scope(profile)
        return {
            "connected": bool(profile.get("access_token") and profile.get("refresh_token")),
            "email": profile.get("email"),
            "client_id": profile.get("client_id"),
            "scopes": list(profile.get("scopes", [])),
            "expires_at": profile.get("expires_at"),
            "plan_usage_enabled": PLAN_SCOPE in profile.get("scopes", []),
        }

    def sign_out(self) -> tuple[bool, bool]:
        """Revoke the renewable session, then clear local tokens.

        Returns (credentials_found, remote_revocation_confirmed). Local tokens are
        cleared even if OpenAI cannot confirm revocation.
        """
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=30):
            profile = self._read_profile()
            if profile is None:
                return False, False

            refresh_token = profile.get("refresh_token")
            client_id = profile.get("client_id")
            revoked = False
            if isinstance(refresh_token, str) and refresh_token and isinstance(client_id, str) and client_id:
                for attempt in range(3):
                    try:
                        response = httpx.post(
                            REVOCATION_ENDPOINT,
                            data={
                                "token": refresh_token,
                                "token_type_hint": "refresh_token",
                                "client_id": client_id,
                            },
                            timeout=10.0,
                        )
                        if response.status_code >= 500:
                            if attempt < 2:
                                time.sleep(0.5 * (2**attempt))
                                continue
                            break
                        response.raise_for_status()
                        revoked = response.status_code == 200
                        break
                    except httpx.RequestError:
                        if attempt < 2:
                            time.sleep(0.5 * (2**attempt))
                    except httpx.HTTPStatusError:
                        break

            for key in ("access_token", "refresh_token", "id_token"):
                profile[key] = ""
            profile["expires_at"] = 0
            self._write_profile(profile)
            return True, revoked

    @staticmethod
    def _validate_id_token(token: str, client_id: str, nonce: str) -> dict[str, Any]:
        try:
            signing_key = jwt.PyJWKClient(JWKS_URL).get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=client_id,
                leeway=5,
                issuer=ISSUER,
                options={"require": ["exp", "iss", "aud", "sub", "nonce"]},
            )
        except Exception as exc:
            raise ChatGPTOAuthError("OpenAI's ID token could not be validated. No credentials were saved.") from exc
        if not secrets.compare_digest(str(claims.get("nonce", "")), nonce):
            raise ChatGPTOAuthError("OpenAI's ID token nonce did not match this authorization attempt.")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise ChatGPTOAuthError("OpenAI's ID token has no subject.")
        return claims

    def _get_or_create_host_id(self) -> str:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._set_directory_permissions()
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=30):
            if self.host_id_path.exists():
                host_id = self.host_id_path.read_text(encoding="utf-8").strip()
                if host_id.startswith("urn:uuid:"):
                    return host_id
            host_id = f"urn:uuid:{uuid.uuid4()}"
            self._atomic_write(self.host_id_path, host_id + "\n")
            return host_id

    def _read_profile(self) -> dict[str, Any] | None:
        if not self.profile_path.exists():
            return None
        try:
            value = json.loads(self.profile_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChatGPTOAuthError("The saved ChatGPT credential file is unreadable. Sign in again.") from exc
        if not isinstance(value, dict):
            raise ChatGPTOAuthError("The saved ChatGPT credential file is invalid.")
        return value

    def _write_profile(self, profile: dict[str, Any]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._set_directory_permissions()
        self._atomic_write(self.profile_path, json.dumps(profile, ensure_ascii=False, indent=2) + "\n")

    @staticmethod
    def _atomic_write(path: Path, value: str) -> None:
        fd, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as temporary_file:
                temporary_file.write(value)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            try:
                os.chmod(temporary_path, 0o600)
            except OSError:
                pass
            os.replace(temporary_path, path)
        finally:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    def _set_directory_permissions(self) -> None:
        try:
            os.chmod(self.config_dir, 0o700)
        except OSError:
            pass

    @staticmethod
    def _ensure_plan_scope(profile: dict[str, Any]) -> None:
        if PLAN_SCOPE not in profile.get("scopes", []):
            raise ChatGPTOAuthError("The saved session does not have ChatGPT plan usage permission. Sign in again.")

    @staticmethod
    def _positive_int(value: Any, *, default: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            return default
        return result if result > 0 else default
