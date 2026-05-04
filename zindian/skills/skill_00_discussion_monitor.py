"""
Skill 00 — Discussion Monitor
Runs BEFORE any other skill and continuously in the background.
Fetches all competition discussions, flags rule changes and bans,
writes to SKILL_STATE.json and reports/compliance_log.md.

Must run:
  - At session start (before Phase 1)
  - Before Skill 07 (feature engineering)
  - Before Skill 17 (final submission)
"""

import os
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

from zindian.paths import CompetitionPaths, resolve_competition_paths

BASE_URL = "https://api.zindi.africa/v1/competitions"

COMPLIANCE_KEYWORDS = [
    "banned", "not allowed", "prohibited", "clarification",
    "correction", "updated", "disqualified", "external data",
    "you must not", "do not use", "removed", "retracted",
    "spatial", "leakage", "forbidden", "violation", "warning",
    "important", "announcement", "rule change", "amended"
]


def fetch_discussions(slug: str, headers: dict) -> list:
    url = f"{BASE_URL}/{slug}/discussions"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_discussion_detail(slug: str, discussion_id: int,
                            headers: dict) -> dict:
    url = f"{BASE_URL}/{slug}/discussions/{discussion_id}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        try:
            return resp.json().get("data", {})
        except Exception:
            return {}
    return {}


def is_compliance_relevant(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in COMPLIANCE_KEYWORDS)


def extract_compliance_flags(discussions: list, slug: str,
                              headers: dict) -> list:
    flagged = []
    for d in discussions:
        title = d.get("title", "")
        published = d.get("published_at", "")[:10]
        discussion_id = d["id"]

        detail = fetch_discussion_detail(slug, discussion_id, headers)
        body = detail.get("body", "")
        comments = detail.get("comments", [])

        title_flagged = is_compliance_relevant(title)
        body_flagged = is_compliance_relevant(body)

        flagged_comments = []
        for c in comments:
            c_text = c.get("body", "")
            if is_compliance_relevant(c_text):
                flagged_comments.append({
                    "author": c.get("user", {}).get("username", "?"),
                    "text": c_text[:300],
                    "date": c.get("created_at", "")[:10]
                })

        if title_flagged or body_flagged or flagged_comments:
            flagged.append({
                "id": discussion_id,
                "title": title,
                "published": published,
                "body_preview": body[:500],
                "flagged_comments": flagged_comments,
                "url": f"https://zindi.africa/competitions/{slug}/discussions/{discussion_id}"
            })

    return flagged


def write_compliance_log(flagged: list, slug: str,
                         all_discussions: list,
                         paths: CompetitionPaths) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    path = paths.reports_dir / "compliance_log.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Competition Compliance Log",
        f"**Competition**: {slug}",
        f"**Last Updated**: {now}",
        f"**Total Discussions**: {len(all_discussions)}",
        f"**Flagged for Compliance**: {len(flagged)}",
        "",
        "---",
        "",
        "## ⚠️ Compliance Flags",
        "",
    ]

    if not flagged:
        lines.append("_No compliance issues detected._")
    else:
        for f in flagged:
            lines += [
                f"### 🚨 {f['title']}",
                f"**Date**: {f['published']}  ",
                f"**URL**: {f['url']}  ",
                "",
                f"{f['body_preview']}",
                "",
            ]
            if f["flagged_comments"]:
                lines.append("**Flagged Comments**:")
                for c in f["flagged_comments"]:
                    lines.append(
                        f"- **{c['author']}** ({c['date']}): {c['text']}"
                    )
            lines += ["", "---", ""]

    lines += [
        "## 📋 All Discussions",
        "",
        "| Date | Title | Comments |",
        "|------|-------|----------|",
    ]
    for d in all_discussions:
        title = d.get("title", "")[:60]
        date = d.get("published_at", "")[:10]
        count = d.get("comments_count", 0)
        lines.append(f"| {date} | {title} | {count} |")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Compliance log -> {path}")


def update_skill_state(flagged: list, all_discussions: list,
                       paths: CompetitionPaths) -> None:
    state_path = paths.state_path
    if not state_path.exists():
        return

    with open(state_path, encoding="utf-8") as f:
        state = json.load(f)

    state["compliance"] = {
        "last_checked": datetime.now(timezone.utc).isoformat(),
        "total_discussions": len(all_discussions),
        "flagged_count": len(flagged),
        "flagged_titles": [f["title"] for f in flagged],
        "agent_must_read": len(flagged) > 0
    }

    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp:
        json.dump(state, tmp, indent=2)
        tmp_path = tmp.name
    os.replace(tmp_path, state_path)
    print(f"✅ {state_path} updated with compliance summary")


def run(slug: str, headers: dict) -> dict:
    """
    Main entry point.
    Call before any other skill and before every submission.
    """
    print(f"\n{'='*60}")
    print(f"SKILL 00 — Discussion Monitor")
    print(f"Competition: {slug}")
    print(f"{'='*60}\n")

    paths = resolve_competition_paths(slug=slug)

    all_discussions = fetch_discussions(slug, headers)
    print(f"Found {len(all_discussions)} discussions")

    flagged = extract_compliance_flags(all_discussions, slug, headers)
    print(f"Flagged {len(flagged)} for compliance review")

    write_compliance_log(flagged, slug, all_discussions, paths)
    update_skill_state(flagged, all_discussions, paths)

    if flagged:
        print("\n⚠️  COMPLIANCE ISSUES DETECTED:")
        for f in flagged:
            print(f"  🚨 {f['title']}")
        print(f"\n-> Read {paths.reports_dir / 'compliance_log.md'} before proceeding")
    else:
        print("\n✅ No compliance issues detected")

    return {
        "status": "WARNING" if flagged else "CLEAR",
        "flagged_count": len(flagged),
        "flagged": flagged,
        "total_discussions": len(all_discussions)
    }
