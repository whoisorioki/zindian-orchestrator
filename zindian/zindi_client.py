"""
ZindiClient — Direct API wrapper for the Zindi platform.
Bypasses the broken select_a_challenge() in the KameniAlexNea package
and talks directly to https://api.zindi.africa/v1/competitions
"""

import os
import requests
from dotenv import load_dotenv
from zindi.user import Zindian

load_dotenv()


class ZindiClient:
    BASE_URL = "https://api.zindi.africa/v1/competitions"

    def __init__(self):
        self._user = Zindian(
            username=os.getenv("ZINDI_USERNAME"),
            fixed_password=os.getenv("ZINDI_PASSWORD"),
        )
        self._auth_token = self._user._Zindian__auth_data["auth_token"]
        self._headers = {**self._user._Zindian__headers, "token": self._auth_token}
        self._challenge_id = None
        print(f"✅ Logged in as: {os.getenv('ZINDI_USERNAME')}")

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
        # Use fixed_index workaround: fetch list, find index, select
        competitions = self.list_competitions(active=False)
        all_ids = [c["id"] for c in competitions]

        if challenge_id not in all_ids:
            # Try fetching all (including closed)
            resp = requests.get(self.BASE_URL, headers=self._headers)
            resp.raise_for_status()
            all_comps = resp.json()["data"]
            all_ids = [c["id"] for c in all_comps]

        if challenge_id in all_ids:
            idx = all_ids.index(challenge_id)
            self._user.select_a_challenge(fixed_index=idx)
            self._challenge_id = challenge_id
            print(f"✅ Selected: {challenge_id}")
        else:
            raise ValueError(f"Competition '{challenge_id}' not found.")

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
            return self._user.remaining_subimissions
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
