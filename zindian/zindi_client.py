"""
ZindiClient — Direct API wrapper for the Zindi platform.
Bypasses the broken select_a_challenge() in the KameniAlexNea package
and talks directly to https://api.zindi.africa/v1/competitions
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

load_dotenv()


class ZindiClient:
    BASE_URL = "https://api.zindi.africa/v1/competitions"

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
            remaining = getattr(cast(Any, self._user), "remaining_subimissions", -1)
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
