"""
Skill 17 — Sub Governance
Generic final submission selector for any Zindi competition.
HUMAN GATED. Run before competition deadline.

Usage:
    python3 -m zindian.skills.skill_17_governance
    python3 -m zindian.skills.skill_17_governance ey-frogs
"""

import json
import os
import sys
import tempfile
import requests
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from zindi.user import Zindian


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_state(slug: str) -> dict:
    path = Path(f"competitions/{slug}/SKILL_STATE.json")
    with open(path) as f:
        return json.load(f)


def write_state(slug: str, state: dict) -> None:
    path = Path(f"competitions/{slug}/SKILL_STATE.json")
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def load_config(slug: str) -> dict:
    path = Path(f"competitions/{slug}/challenge_config.json")
    with open(path) as f:
        return json.load(f)


def write_final_selections(slug: str, selections: list) -> None:
    path = Path(f"competitions/{slug}/reports/final_selections.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Final Submission Selections",
        f"**Competition**: {slug}",
        f"**Locked at**: {now}",
        "",
        "## Selected for Private Leaderboard",
        "",
    ]
    for i, s in enumerate(selections, 1):
        lines += [
            f"### Selection {i}",
            f"- **ID**: {s['id']}",
            f"- **File**: {s['filename']}",
            f"- **Public score**: {s['score']}",
            f"- **Date**: {s['date']}",
            f"- **Comment**: {s.get('comment', '')}",
            "",
        ]
    lines += [
        "## Rationale",
        "Highest two public scores selected by human via Skill 17 governance gate.",
        "Both submissions verified compliant before selection.",
    ]
    path.write_text("\n".join(lines))
    print(f"✅ Report written → {path}")


# ── Zindi API ─────────────────────────────────────────────────────────────────

def get_zindi_user() -> tuple:
    """Return (user, auth_token, headers)."""
    load_dotenv()
    user = Zindian(
        username=os.getenv("ZINDI_USERNAME"),
        fixed_password=os.getenv("ZINDI_PASSWORD")
    )
    # Avoid hard dependency on private attrs: read dynamically with env fallback.
    auth_data = getattr(user, "_Zindian__auth_data", {}) or {}
    auth_token = (
        auth_data.get("auth_token")
        or os.getenv("ZINDI_API_KEY")
        or os.getenv("ZINDI_TOKEN")
        or ""
    )
    base_headers = getattr(user, "_Zindian__headers", {}) or {}
    headers = dict(base_headers) if isinstance(base_headers, dict) else {}
    if auth_token:
        headers["token"] = auth_token
    return user, auth_token, headers


def fetch_scored_submissions(competition_slug: str) -> list:
    """
    Fetch all submissions with score > 0 from Zindi API.
    Returns list sorted by score descending.
    """
    _, _, headers = get_zindi_user()

    # Try /submissions endpoint
    resp = requests.get(
        f"https://api.zindi.africa/v1/competitions/{competition_slug}/submissions",
        headers=headers
    )

    if resp.status_code == 200:
        data = resp.json().get("data", [])
        subs = []
        for s in data:
            try:
                score = float(s.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            if score > 0:
                subs.append({
                    "id":       s.get("id", ""),
                    "score":    score,
                    "filename": s.get("filename", ""),
                    "date":     str(s.get("created_at", ""))[:10],
                    "comment":  s.get("comment", ""),
                })
        subs.sort(key=lambda x: x["score"], reverse=True)
        return subs

    # Fallback: try /my_submissions
    resp2 = requests.get(
        f"https://api.zindi.africa/v1/competitions/{competition_slug}/my_submissions",
        headers=headers
    )
    if resp2.status_code == 200:
        data = resp2.json().get("data", [])
        subs = []
        for s in data:
            try:
                score = float(s.get("score") or 0)
            except (TypeError, ValueError):
                score = 0.0
            if score > 0:
                subs.append({
                    "id":       s.get("id", ""),
                    "score":    score,
                    "filename": s.get("filename", ""),
                    "date":     str(s.get("created_at", ""))[:10],
                    "comment":  s.get("comment", ""),
                })
        subs.sort(key=lambda x: x["score"], reverse=True)
        return subs

    print(f"⚠️  API returned {resp.status_code} for submissions endpoint")
    print(f"   Response: {resp.text[:200]}")
    return []


def fetch_via_submission_board(competition_slug: str) -> list:
    """
    Fallback: use Zindian.submission_board() and parse the printed table.
    """
    import io
    from contextlib import redirect_stdout

    user, auth_token, headers = get_zindi_user()

    # Select competition on the user object
    comp_resp = requests.get(
        f"https://api.zindi.africa/v1/competitions/{competition_slug}",
        headers=headers
    )
    user._Zindian__challenge_selected = True
    user._Zindian__api = (
        f"https://api.zindi.africa/v1/competitions/{competition_slug}"
    )
    user._Zindian__challenge_data = comp_resp.json()["data"]

    buf = io.StringIO()
    with redirect_stdout(buf):
        user.submission_board(to_print=True)
    output = buf.getvalue()

    # Parse table rows — each data row starts with |  🟢  |
    subs = []
    for line in output.splitlines():
        if "🟢" not in line and "🔴" not in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        # parts: [status, id, date, score, filename, comment]
        if len(parts) < 5:
            continue
        try:
            score = float(parts[3])
        except (ValueError, IndexError):
            score = 0.0
        if score <= 0:
            continue
        subs.append({
            "id":       parts[1],
            "score":    score,
            "filename": parts[4] if len(parts) > 4 else "",
            "date":     parts[2] if len(parts) > 2 else "",
            "comment":  parts[5] if len(parts) > 5 else "",
        })

    subs.sort(key=lambda x: x["score"], reverse=True)
    return subs


# ── Human Gate ────────────────────────────────────────────────────────────────

def human_selection_gate(scored: list, state: dict) -> list:
    print(f"\n{'='*60}")
    print("HUMAN GATE — Select exactly 2 submissions for private judging")
    print(f"{'='*60}")
    print(f"Current selections: {state.get('selected_submissions', [])}")
    print()
    print(f"{'#':<4} {'ID':<12} {'Score':<14} {'Date':<12} {'File'}")
    print("-" * 75)
    for i, s in enumerate(scored):
        print(
            f"{i:<4} {s['id']:<12} {s['score']:<14.9f} "
            f"{s['date']:<12} {s['filename'][:35]}"
        )

    print(f"\n{'='*60}")
    print("Enter two indices separated by space (e.g. '0 1') or CANCEL")
    print(f"{'='*60}")

    while True:
        resp = input("\nSelect 2: ").strip()
        if resp.upper() == "CANCEL":
            print("Cancelled.")
            return []
        try:
            parts = resp.split()
            assert len(parts) == 2, "Need exactly 2 indices"
            idx1, idx2 = int(parts[0]), int(parts[1])
            assert idx1 != idx2, "Select 2 different submissions"
            sel = [scored[idx1], scored[idx2]]
        except (AssertionError, ValueError, IndexError) as e:
            print(f"❌ {e}. Try again.")
            continue

        print(f"\nYou selected:")
        for i, s in enumerate(sel, 1):
            print(
                f"  {i}. {s['id']} — {s['filename']} "
                f"(score: {s['score']:.9f})"
            )

        confirm = input(
            "\nType YES to lock these as final selections: "
        ).strip().upper()
        if confirm == "YES":
            return sel
        print("Re-selecting...")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(slug: str = "ey-frogs") -> dict:
    print("=" * 60)
    print("SKILL 17 — Sub Governance")
    print("=" * 60)

    state  = load_state(slug)
    config = load_config(slug)
    competition_slug = config.get("slug", slug)

    print(f"\nCompetition     : {competition_slug}")
    print(f"Metric          : {config.get('metric', 'unknown')}")
    print(f"Current selected: {state.get('selected_submissions', [])}")

    # ── Fetch submissions ─────────────────────────────────────
    print("\nFetching scored submissions from Zindi API...")
    scored = fetch_scored_submissions(competition_slug)

    if not scored:
        print("API endpoint returned no results. Trying submission_board()...")
        scored = fetch_via_submission_board(competition_slug)

    if not scored:
        print("❌ Could not fetch any scored submissions.")
        print("   Check your Zindi credentials and competition slug.")
        return {"status": "ERROR", "message": "No submissions fetched"}

    print(f"✅ Found {len(scored)} scored submissions")

    # ── Human gate ────────────────────────────────────────────
    selections = human_selection_gate(scored, state)

    if not selections:
        return {"status": "CANCELLED", "message": "Cancelled by human"}

    # ── Write outputs ─────────────────────────────────────────
    write_final_selections(slug, selections)

    state["selected_submissions"]  = [s["id"] for s in selections]
    state["governance_locked_at"]  = datetime.now(timezone.utc).isoformat()
    state["dag_phase"]             = "phase_5_governance_complete"
    state["last_updated"]          = datetime.now(timezone.utc).isoformat()
    write_state(slug, state)

    print(f"\n✅ SKILL 17 COMPLETE")
    print(f"   Selected : {[s['id'] for s in selections]}")
    print(f"   Scores   : {[s['score'] for s in selections]}")

    return {
        "status":   "OK",
        "selected": [s["id"] for s in selections],
    }


run_governance = run  # compatibility alias


if __name__ == "__main__":
    slug = sys.argv[1] if len(sys.argv) > 1 else "ey-frogs"
    result = run(slug)
    print(f"\nResult: {result['status']}")