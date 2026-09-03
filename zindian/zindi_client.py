"""
ZindiClient — Direct API wrapper for the Zindi platform.
Bypasses the broken select_a_challenge() in the KameniAlexNea package
and talks directly to https://api.zindi.world/v1/competitions
"""

import os
import requests
from dotenv import load_dotenv
import sys
from typing import Any, cast


# Dynamically resolve Zindian to bypass the local shadow package 'zindi/' in the repository root,
# except when running offline/unit tests where the stub is required.
def _get_zindian_class():
    if os.environ.get("ZINDIAN_DISABLE_NETWORK") == "1" or "pytest" in sys.modules:
        from zindi.user import Zindian

        return Zindian

    saved_path = list(sys.path)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path = [p for p in sys.path if os.path.abspath(p) != repo_root and p != ""]
    try:
        from zindi.user import Zindian

        return Zindian
    except ImportError:
        sys.path = saved_path
        from zindi.user import Zindian

        return Zindian
    finally:
        sys.path = saved_path


Zindian = _get_zindian_class()

def _harden_zindi_upload_timeouts(connect_timeout: float = 30.0, read_timeout: float = 300.0) -> None:
    """
    Patch the vendored zindi package's upload() to enforce HTTP timeouts.

    The upstream `zindi.utils.upload()` (v0.0.4) calls `requests.post(...)`
    with NO timeout, so a stalled Zindi API connection (accepted bytes but
    no response) blocks skill_16's submit indefinitely. We rebind
    `zindi.platform_api.upload` (the exact symbol platform_api resolved via
    `from zindi.utils import upload` at import time) with a copy of the same
    MultipartEncoderMonitor logic plus `timeout=(connect, read)`.

    Idempotent: skips if the patch is already applied.
    """
    import zindi.platform_api as _platform_api

    if getattr(_platform_api.upload, "_zindian_timeout_patched", False):
        return

    from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
    from tqdm import tqdm

    def _upload_with_timeout(filepath, comment, url, headers):
        filename = (os.sep).join(filepath.split(os.sep)[-2:])
        encoder = MultipartEncoder(
            {"file": (filename, open(filepath, "rb"), "text/plain"), "comment": comment}
        )
        with tqdm(
            desc=f"Submit {filename}",
            total=encoder.len,
            ncols=100,
            unit="o",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            multipart_monitor = MultipartEncoderMonitor(
                encoder,
                lambda monitor: progress_bar.update(monitor.bytes_read - progress_bar.n),
            )
            req_headers = {**headers, "Content-Type": multipart_monitor.content_type}
            response = requests.post(
                url,
                data=multipart_monitor,
                params={"auth_token": headers["auth_token"]},
                headers=req_headers,
                timeout=(connect_timeout, read_timeout),
            )
        return response

    _upload_with_timeout._zindian_timeout_patched = True  # type: ignore[attr-defined]
    _platform_api.upload = _upload_with_timeout
    _upload_with_timeout._zindian_timeout_patched = True  # type: ignore[attr-defined]
    _platform_api.upload = _upload_with_timeout


# The canonical Zindi API base. The vendored zindi package (v0.0.4) still
# hardcodes the deprecated `api.zindi.africa` domain in six places; the
# platform migrated to `zindi.world`. All endpoints are normalized here.
ZINDI_WORLD_API = "https://api.zindi.world/v1/competitions"


def _normalize_zindi_endpoints(user: Any = None) -> None:
    """
    Force every vendored-zindi endpoint onto api.zindi.world (idempotent).

    Class-level patches (apply to future instances, incl. signin):
      - PlatformAPI.signin            -> posts to world auth endpoint
      - PlatformAPI._auth_headers     -> rewrites `current_url` headers
                                        from zindi.africa to zindi.world
    Instance-level rewrites (cover the already-constructed session):
      - Zindian._Zindian__base_api    -> world competitions base
      - Zindian._Zindian__api_client.base_api -> world competitions base
    """
    import zindi.platform_api as _platform_api

    if not getattr(_platform_api, "_zindian_world_normalized", False):
        # signin() uses an inline literal URL (not base_api) — reimplement.
        def _signin_world(self, username, password):
            response = requests.post(
                "https://api.zindi.world/v1/auth/signin",
                data={"username": username, "password": password},
                headers=self.default_headers,
            )
            data = self._response_data(response)
            return self._raise_on_errors(data)

        _platform_api.ZindiPlatformAPI.signin = _signin_world

        # Normalize referer-style `current_url` headers that still point at
        # the deprecated domain (leaderboard / participations calls).
        _orig_auth_headers = _platform_api.ZindiPlatformAPI._auth_headers

        def _auth_headers_world(self, auth_token, current_url=None):
            if isinstance(current_url, str) and "zindi.africa" in current_url:
                current_url = current_url.replace("zindi.africa", "zindi.world")
            return _orig_auth_headers(self, auth_token, current_url)

        _platform_api.ZindiPlatformAPI._auth_headers = _auth_headers_world
        _platform_api._zindian_world_normalized = True

    if user is not None:
        # Rewrite the already-constructed session's base endpoints.
        user._Zindian__base_api = ZINDI_WORLD_API
        api_client = getattr(user, "_Zindian__api_client", None)
        if api_client is not None and hasattr(api_client, "base_api"):
            api_client.base_api = ZINDI_WORLD_API


load_dotenv()

class ZindiClient:
    BASE_URL = "https://api.zindi.world/v1/competitions"

    def __init__(self):
        self._user = Zindian(
            username=os.getenv("ZINDI_USERNAME"),
            fixed_password=os.getenv("ZINDI_PASSWORD"),
        )
        auth_data = cast(Any, self._user)._Zindian__auth_data
        self._auth_token = auth_data["auth_token"]
        self._headers = {
            **cast(Any, self._user)._Zindian__headers,
            "token": self._auth_token,
        }
        self._challenge_id = None
        # Normalize deprecated api.zindi.africa endpoints -> api.zindi.world
        # BEFORE any session use (select_challenge builds URLs from base_api).
        _normalize_zindi_endpoints(self._user)
        print(f"  [OK] Logged in as: {os.getenv('ZINDI_USERNAME')}")

    # ── Competition Discovery ──────────────────────────────────────

    def list_competitions(
        self, kind="competition", active=True, beginner_friendly=False
    ) -> list:
        """
        Fetch competitions directly from the API.
        Returns a list of competition dicts with keys:
          id, title, end_time, is_beginner_friendly, kind,
          open, participations_count, reward
        """
        params = {}
        if active:
            params["active"] = "true"

        resp = requests.get(self.BASE_URL, headers=self._headers, params=params)
        resp.raise_for_status()

        competitions = resp.json()["data"]

        # Filter by kind
        competitions = [c for c in competitions if c.get("kind") == kind]

        # Optionally filter beginner friendly
        if beginner_friendly:
            competitions = [c for c in competitions if c.get("is_beginner_friendly")]

        return competitions

    def print_competitions(self, competitions: list):
        """Print a numbered list of competitions for selection."""
        print(f"\n{'#':<4} {'ID':<70} {'Beginner':<10} {'Ends'}")
        print("-" * 110)
        for i, c in enumerate(competitions):
            end = c.get("end_time", "")[:10]
            beginner = "✅" if c.get("is_beginner_friendly") else "  "
            restricted = "🔒" if c.get("is_access_restricted") else "  "
            print(f"{i:<4} {c['id']:<70} {beginner:<10} {end} {restricted}")

    def select_competition(self, challenge_id: str):
        """
        Select a competition by its slug/id.
        Also selects it on the Zindian object for submit() to work.
        """
        # Select directly by challenge_id instead of the fragile fixed_index workaround
        res = self._user.select_a_challenge(challenge_id=challenge_id)
        if isinstance(res, dict) and res.get("challenge") is None:
            raise ValueError(
                f"Competition '{challenge_id}' not found: {res.get('message')}"
            )
        self._challenge_id = self._user.which_challenge
        print(f"✅ Selected: {self._challenge_id}")

    # ── Competition Info ───────────────────────────────────────────

    def get_competition_details(self, challenge_id: str) -> dict:
        """Fetch full details for a specific competition."""
        url = f"{self.BASE_URL}/{challenge_id}"
        resp = requests.get(url, headers=self._headers)
        resp.raise_for_status()
        return resp.json().get("data", {})

    # ── Submission ─────────────────────────────────────────────────

    @property
    def remaining_submissions(self) -> int:
        """Check submission budget before submitting."""
        try:
            user = cast(Any, self._user)
            remaining = getattr(user, "remaining_submissions", None)
            if remaining is None:
                remaining = getattr(user, "remaining_subimissions", -1)
            return int(remaining) if remaining is not None else -1
        except Exception:
            return -1  # Unknown — do not block, but log warning

    def submit(self, filepath: str, comment: str) -> dict:
        """
        Submit a CSV file to Zindi.
        Always checks remaining submissions first.
        Comment must follow: branch:X|oof_rmse:X|features:N|calib:X
        """
        remaining = self.remaining_submissions
        if remaining == 0:
            raise RuntimeError(
                "❌ Submission blocked: daily limit reached (remaining=0)"
            )

        if remaining == -1:
            print("⚠️  Warning: could not verify remaining submissions. Proceeding.")

        print(f"📤 Submitting: {filepath}")
        print(f"📝 Comment: {comment}")
        print(f"📊 Remaining before submit: {remaining}")

        # Harden the vendored uploader with HTTP timeouts before the call
        # (upstream zindi 0.0.4 posts with no timeout and can hang forever).
        _harden_zindi_upload_timeouts()

        self._user.submit(filepaths=[filepath], comments=[comment])

        rank = self._user.my_rank
        print(f"✅ Submitted. Current rank: {rank}")
        return {"filepath": filepath, "comment": comment, "rank": rank}

    # ── Leaderboard ────────────────────────────────────────────────

    def leaderboard(self, per_page: int = 20) -> None:
        """Print current leaderboard."""
        self._user.leaderboard(to_print=True)

    def my_rank(self) -> int:
        """Return current rank."""
        return self._user.my_rank

    # ── Dataset ────────────────────────────────────────────────────

    def download_dataset(self, destination: str = "data/raw") -> list:
        """Download competition dataset to destination folder."""
        os.makedirs(destination, exist_ok=True)
        return self._user.download_dataset(
            destination=destination, make_destination=True
        )


def _structured_comment(branch: str, oof_rmse: float, features: int, calib: str) -> str:
    """Format structured comment for Zindi submissions."""
    return f"branch:{branch}|oof_rmse:{oof_rmse:.6f}|features:{features}|calib:{calib}"
