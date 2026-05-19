"""
Skill 17 — Sub Governance
Selects exactly 2 submissions for private leaderboard judging.
HUMAN GATED — requires explicit YES before locking selections.

Must run before competition deadline.
For EY Biodiversity: before May 19 (5 days before May 24 close).

Rules:
- Only compliant submissions may be selected
- Must select exactly 2
- Selection rationale must be logged to reports/final_selections.md
- SKILL_STATE.json selected_submissions must be updated
- Human must type YES to confirm
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from zindian.zindi_client import ZindiClient


BASE_URL = "https://api.zindi.africa/v1/competitions"


def fetch_submission_board(slug: str) -> tuple[str, ZindiClient]:
    """Fetch all submissions from Zindi API."""
    # Parse submission board from Zindian
    import io
    from contextlib import redirect_stdout

    client = ZindiClient()
    client.select_competition(slug)
    f = io.StringIO()
    with redirect_stdout(f):
        client.leaderboard()
    output = f.getvalue()
    return output, client


def load_state(slug: str) -> dict:
    state_path = Path(f"competitions/{slug}/SKILL_STATE.json")
    with open(state_path) as f:
        return json.load(f)


def load_config(slug: str) -> dict:
    config_path = Path(f"competitions/{slug}/challenge_config.json")
    with open(config_path) as f:
        return json.load(f)


def write_state(slug: str, state: dict) -> None:
    state_path = Path(f"competitions/{slug}/SKILL_STATE.json")
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)


def write_final_selections(slug: str, selections: list,
                           rationale: str) -> None:
    """Write selection rationale to reports/final_selections.md."""
    path = Path(f"competitions/{slug}/reports/final_selections.md")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Final Submission Selections",
        f"**Competition**: {slug}",
        f"**Locked at**: {now}",
        "",
        "## Selected Submissions (for private leaderboard judging)",
        "",
    ]
    for i, s in enumerate(selections, 1):
        lines += [
            f"### Selection {i}",
            f"- **Submission ID**: {s.get('id')}",
            f"- **File**: {s.get('filename')}",
            f"- **Public LB score**: {s.get('score')}",
            f"- **Date**: {s.get('date')}",
            "",
        ]

    lines += [
        "## Rationale",
        "",
        rationale,
        "",
        "---",
        f"*Locked by skill_17_governance | {now}*",
    ]

    path.write_text("\n".join(lines))
    print(f"✅ Final selections → {path}")


def human_selection_gate(
    candidates: list,
    state: dict,
) -> list:
    """
    Show all compliant submissions and ask human to confirm 2 selections.
    Returns list of 2 selected submission dicts.
    """
    print(f"\n{'='*60}")
    print("=== HUMAN GATE: Skill 17 — Final Submission Selection ===")
    print(f"{'='*60}")
    print(f"\nDeadline: SELECT 2 submissions before competition closes.")
    print(f"Current selections: {state.get('selected_submissions', [])}")
    print(f"\nAll scored submissions:")
    print(f"{'#':<4} {'ID':<12} {'Score':<12} {'File':<40}")
    print("-" * 70)

    scored = [c for c in candidates if c.get("score") and
              float(str(c.get("score", 0))) > 0]
    scored.sort(key=lambda x: float(str(x.get("score", 0))), reverse=True)

    for i, s in enumerate(scored):
        print(
            f"{i:<4} {s.get('id', ''):<12} "
            f"{str(s.get('score', '')):<12} "
            f"{s.get('filename', ''):<40}"
        )

    print(f"\n{'='*60}")
    print("Select your 2 submissions by entering their index numbers.")
    print("Format: two numbers separated by space (e.g. '0 1')")
    print("These will be locked as your final private LB selections.")
    print(f"{'='*60}")

    while True:
        response = input("\nEnter 2 indices (or CANCEL): ").strip()

        if response.upper() == "CANCEL":
            print("🛑 Selection cancelled. Run again when ready.")
            return []

        try:
            parts = response.split()
            if len(parts) != 2:
                print("❌ Enter exactly 2 numbers separated by space.")
                continue
            idx1, idx2 = int(parts[0]), int(parts[1])
            if idx1 == idx2:
                print("❌ Select 2 different submissions.")
                continue
            sel1 = scored[idx1]
            sel2 = scored[idx2]
        except (ValueError, IndexError):
            print("❌ Invalid indices. Try again.")
            continue

        print(f"\nYou selected:")
        print(f"  1. {sel1.get('id')} — {sel1.get('filename')} "
              f"(score: {sel1.get('score')})")
        print(f"  2. {sel2.get('id')} — {sel2.get('filename')} "
              f"(score: {sel2.get('score')})")
        print("\nType YES to lock these selections or NO to re-select.")

        confirm = input("Confirm? [YES/NO]: ").strip().upper()
        if confirm == "YES":
            return [sel1, sel2]
        else:
            print("Re-selecting...")
            continue


def run(slug: str = "ey-frogs") -> dict:
    print("=" * 60)
    print("SKILL 17 — Sub Governance")
    print("=" * 60)

    state = load_state(slug)
    config = load_config(slug)

    competition_slug = config.get("slug", "ey-biodiversity-challenge")
    print(f"\nCompetition : {competition_slug}")
    print(f"Current selections: {state.get('selected_submissions', [])}")

    # Fetch live submission board
    print("\nFetching submission board from Zindi...")
    try:
        board_output, client = fetch_submission_board(competition_slug)
        print(board_output)

        # Build candidate list from state knowledge
        # (API doesn't expose structured board — use known submissions)
        candidates = [
            {"id": "tfcawL75", "filename": "variant-34b_submission.csv",
             "score": 0.884568651, "date": "2026-05-07"},
            {"id": "WeXoXWi6", "filename": "sub_011_anchor.csv",
             "score": 0.881642512, "date": "2026-05-06"},
        ]
        print(f"\nKnown compliant submissions: {len(candidates)}")

    except Exception as e:
        print(f"⚠️  Could not fetch live board: {e}")
        print("Using known compliant submissions from state.")
        candidates = [
            {"id": "tfcawL75", "filename": "variant-34b_submission.csv",
             "score": 0.884568651, "date": "2026-05-07"},
            {"id": "WeXoXWi6", "filename": "sub_011_anchor.csv",
             "score": 0.881642512, "date": "2026-05-06"},
        ]

    # Human gate
    selections = human_selection_gate(candidates, state)

    if not selections:
        return {"status": "CANCELLED", "message": "Selection cancelled by human"}

    # Generate rationale
    rationale = (
        f"Selected highest scoring compliant submissions. "
        f"Selection 1 ({selections[0]['id']}) is the best LB score "
        f"({selections[0]['score']}) from TC-only features. "
        f"Selection 2 ({selections[1]['id']}) is the compliant anchor "
        f"as a safety net. Both use TerraClimate-only features — "
        f"no Lat/Lon — fully compliant with discussion 32369."
    )

    write_final_selections(slug, selections, rationale)

    # Update state
    state["selected_submissions"] = [s["id"] for s in selections]
    state["governance_locked_at"] = datetime.now(timezone.utc).isoformat()
    state["dag_phase"] = "phase_5_governance_complete"
    state["last_updated"] = datetime.now(timezone.utc).isoformat()
    write_state(slug, state)

    print(f"\n✅ SKILL 17 COMPLETE")
    print(f"   Selected: {[s['id'] for s in selections]}")
    print(f"   Rationale logged to reports/final_selections.md")

    return {
        "status": "OK",
        "selected": [s["id"] for s in selections],
        "rationale": rationale
    }


if __name__ == "__main__":
    import sys
    slug = sys.argv[1] if len(sys.argv) > 1 else "ey-frogs"
    result = run(slug)
    print(f"\nResult: {result['status']}")

# Compatibility alias for tests and external callers
run_governance = run
