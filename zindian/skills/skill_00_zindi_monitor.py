"""
Skill 00 — Zindi Monitor (upgraded from Discussion Monitor)
Runs at session start, before Skill 07, and before Skill 17.

Monitors:
  1. Competition page — metric, rules, data files, external data policy
  2. Discussion board — compliance flags, rule changes, bans
  3. Leaderboard — current rank, gap to top, top 10 scores
  4. Submission board — all your submissions with scores

Feeds into:
  - SKILL_STATE.json (compliance, rank, metric confirmation)
  - reports/compliance_log.md (human-readable audit)
  - reports/zindi_monitor.json (machine-readable for Skill 03)
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from zindian.config import ChallengeConfig
from zindian.paths import resolve_competition_paths
from zindian.state import SkillStateStore
from zindian.zindi_client import ZindiClient

BASE_URL = "https://api.zindi.africa/v1/competitions"

# Keywords that trigger compliance flag
COMPLIANCE_KEYWORDS = [
    "banned", "not allowed", "prohibited", "clarification",
    "correction", "updated", "disqualified", "external data",
    "you must not", "do not use", "removed", "retracted",
    "spatial", "leakage", "forbidden", "violation", "warning",
    "important", "announcement", "rule change", "amended",
    "worldclim", "elevation", "srtm", "gbif", "external",
    "additional data", "third party", "outside data",
]

# External data sources to check for in discussions
EXTERNAL_DATA_SOURCES = [
    "worldclim", "srtm", "elevation", "gbif", "modis",
    "sentinel", "landsat", "copernicus", "era5", "chelsa",
    "bioclim", "openstreetmap", "osm", "wc2", "dem",
]


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    from dotenv import load_dotenv
    load_dotenv(override=True)
    token = os.getenv("ZINDI_API_KEY") or os.getenv("ZINDI_TOKEN") or os.getenv("ZINDI_PASSWORD")
    return {"token": token} if token else {}


# ── Section 1: Competition Page ───────────────────────────────────────────────

def scrape_competition_page(slug: str) -> dict:
    """
    Use playwright to scrape the Zindi competition page.
    Generalizable — works for any Zindi competition slug.
    Extracts: metric, rules, evaluation section, prize, deadline.
    """
    from playwright.sync_api import sync_playwright
    import re

    base = f"https://zindi.africa/competitions/{slug}"
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page    = browser.new_page()

        # ── Main competition page ──────────────────────────────
        print(f"  Scraping: {base}")
        page.goto(base, wait_until="networkidle", timeout=30000)
        full_text = page.inner_text("body").lower()

        # Metric detection — Zindi standard evaluation section phrases
        metric = None
        metric_patterns = [
            (r"evaluation metric.*?is\s+([a-z0-9_ ]+score|rmse|mae|auc|accuracy|log.?loss)", "group"),
            (r"scored using\s+([a-z0-9_ ]+score|rmse|mae|auc|accuracy|log.?loss)", "group"),
            (r"error metric.*?(f1|rmse|mae|auc|accuracy|log.?loss|f-1)", "group"),
        ]
        for pattern, _ in metric_patterns:
            m = re.search(pattern, full_text)
            if m:
                raw = m.group(1).strip()
                if "f1" in raw or "f-1" in raw:
                    metric = "f1_score"
                elif "rmse" in raw:
                    metric = "rmse"
                elif "mae" in raw:
                    metric = "mae"
                elif "log" in raw:
                    metric = "log_loss"
                elif "auc" in raw:
                    metric = "auc"
                elif "accuracy" in raw:
                    metric = "accuracy"
                break

        # Fallback: look for explicit metric name in evaluation section
        if not metric:
            if "f1 score" in full_text or "f-1 score" in full_text:
                metric = "f1_score"
            elif "root mean squared" in full_text or "rmse" in full_text:
                metric = "rmse"
            elif "mean absolute" in full_text or " mae" in full_text:
                metric = "mae"
            elif "log loss" in full_text or "logloss" in full_text:
                metric = "log_loss"
            elif "area under" in full_text or " auc " in full_text:
                metric = "auc"
            elif "accuracy" in full_text:
                metric = "accuracy"

        # Use probabilities — Zindi standard phrase
        use_probabilities = (
            "do not set thresholds" in full_text
            or "raw probabilities" in full_text
            or "if the error metric requires probabilities" in full_text
        )

        # External data — Zindi standard phrase
        external_banned = (
            "you may use only the datasets provided" in full_text
            or "only the datasets provided for this challenge" in full_text
            or "no external data" in full_text
        )

        # AutoML — Zindi standard phrase
        automl_banned = (
            "automl are not permitted" in full_text
            or "automated machine learning" in full_text
        )

        # Code review tier
        code_review = None
        cr_match = re.search(r"top (\d+) on the private leaderboard", full_text)
        if cr_match:
            code_review = f"top_{cr_match.group(1)}"

        # Must select submissions
        must_select_2 = (
            "choose 2 submissions" in full_text
            or "select 2 submissions" in full_text
        )

        # Daily and total limits
        daily_limit, total_limit = 10, 300
        dm = re.search(r"maximum of (\d+) submissions per day", full_text)
        tm = re.search(r"maximum of (\d+) submissions for this challenge", full_text)
        if dm:
            daily_limit = int(dm.group(1))
        if tm:
            total_limit = int(tm.group(1))

        # Public/private split
        public_split = 20
        pm = re.search(r"public leaderboard includes approximately (\d+)%", full_text)
        if pm:
            public_split = int(pm.group(1))

        # Max team size
        team_size = 4
        ts = re.search(r"team of up to (?:a maximum of )?(\d+) people", full_text)
        if ts:
            team_size = int(ts.group(1))

        # Seed required
        seed_required = "always set the seed" in full_text

        # Prize amounts
        prizes = {}
        for place, label in [(1,"1st"),(2,"2nd"),(3,"3rd")]:
            pm2 = re.search(rf"{place}(?:st|nd|rd)\s+prize.*?\$\s*([\d,]+)", full_text)
            if pm2:
                prizes[label] = int(pm2.group(1).replace(",",""))

        # Deadline
        deadline = None
        dl = re.search(r"close[sd]?\s+(?:on|at)?\s*(may|june|july|august|september|october)\s+(\d+),?\s*(\d{4})", full_text)
        if dl:
            deadline = f"{dl.group(1).capitalize()} {dl.group(2)}, {dl.group(3)}"

        results = {
            "metric":            metric,
            "use_probabilities": use_probabilities,
            "external_banned":   external_banned,
            "automl_banned":     automl_banned,
            "code_review_tier":  code_review,
            "daily_limit":       daily_limit,
            "total_limit":       total_limit,
            "public_split_pct":  public_split,
            "team_size":         team_size,
            "must_select_2":     must_select_2,
            "seed_required":     seed_required,
            "prizes":            prizes,
            "deadline":          deadline,
        }

        # ── Data page ──────────────────────────────────────────
        print(f"  Scraping: {base}/data")
        try:
            page.goto(f"{base}/data", wait_until="networkidle", timeout=30000)
            data_text = page.inner_text("body").lower()
            # Check for external data sources mentioned on data page
            external_sources = [
                "worldclim","srtm","elevation","gbif","modis","sentinel",
                "landsat","era5","chelsa","bioclim","dem","openstreetmap",
            ]
            results["external_sources_on_data_page"] = [
                s for s in external_sources if s in data_text
            ]
        except Exception as e:
            print(f"  ⚠️  Data page scrape failed: {e}")
            results["external_sources_on_data_page"] = []

        browser.close()

    return results


def fetch_competition_intel(slug: str, headers: dict, config=None) -> dict:
    """
    Fetch competition intelligence.
    Primary: playwright scrape of competition page (generalizable).
    Fallback: challenge_config.json if scrape fails.
    Also pulls datafiles from Zindi API.
    """
    # Get datafiles from API (reliable)
    try:
        url  = f"{BASE_URL}/{slug}"
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        api_data  = resp.json().get("data", {})
        datafiles = [f["filename"] for f in api_data.get("datafiles", [])]
        end_time  = api_data.get("end_at", "")
        prize     = api_data.get("prize", "")
        subtitle  = api_data.get("subtitle", "")
    except Exception as e:
        print(f"  ⚠️  API fetch failed: {e}")
        datafiles, end_time, prize, subtitle = [], "", "", ""

    # Playwright scrape
    try:
        scraped = scrape_competition_page(slug)
        print(f"  ✅ Playwright scrape successful")
        source = "playwright"
    except Exception as e:
        print(f"  ⚠️  Playwright scrape failed: {e}")
        scraped = {}
        source  = "fallback"

    # Config always overrides scraped values for competition-specific rules.
    # Zindi generic T&Cs (scraped from page) may differ from competition rules.
    # challenge_config.json is populated by Skill 02 from the actual competition
    # page and discussion board — it is the authoritative source.
    if config:
        scraped.setdefault("competition_intel", {})
        if config.get("metric"):
            scraped["metric"] = config.get("metric")
        if config.get("use_probabilities") is not None:
            scraped["use_probabilities"] = config.get("use_probabilities")
        if config.get("allowed_external_data") is not None:
            scraped["competition_intel"]["external_banned"] = not config.get("allowed_external_data")
        if config.get("automl_permitted") is not None:
            scraped["competition_intel"]["automl_banned"] = not config.get("automl_permitted")
        if config.get("code_review_tier"):
            scraped["code_review_tier"] = config.get("code_review_tier")
        if config.get("daily_limit"):
            scraped["daily_limit"] = config.get("daily_limit")
        if config.get("total_limit"):
            scraped["total_limit"] = config.get("total_limit")
        if config.get("max_team_size"):
            scraped["team_size"] = config.get("max_team_size")
        if config.get("public_split_pct"):
            scraped["public_split_pct"] = config.get("public_split_pct")
        source = source + "+config_override"
        print(f"  ✅ Competition-specific config applied over generic Zindi rules")

    # Check for external data hints in provided filenames
    external_sources = [
        "worldclim","srtm","elevation","gbif","modis","sentinel",
        "landsat","era5","chelsa","bioclim","dem",
    ]
    external_hints = [
        f for f in datafiles
        if any(src in f.lower() for src in external_sources)
    ]

    banned_features  = config.get("banned_features", []) if config else []
    compliance_notes = config.get("compliance_notes", []) if config else []

    return {
        **scraped,
        "datafiles":         datafiles,
        "external_hints":    external_hints,
        "banned_features":   banned_features,
        "compliance_notes":  compliance_notes,
        "end_time":          end_time,
        "prize":             prize,
        "subtitle":          subtitle,
        "intel_source":      source,
    }



def fetch_discussions(slug: str, headers: dict) -> list:
    url  = f"{BASE_URL}/{slug}/discussions"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_discussion_detail(slug: str, did: int, headers: dict) -> dict:
    url  = f"{BASE_URL}/{slug}/discussions/{did}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        try:
            return resp.json().get("data", {})
        except Exception:
            return {}
    return {}


def is_compliance_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in COMPLIANCE_KEYWORDS)


def scan_external_data_mentions(text: str) -> list[str]:
    t = text.lower()
    return [src for src in EXTERNAL_DATA_SOURCES if src in t]


def load_superseded_flags(paths) -> dict[str, bool]:
    """
    Preserve manual superseded overrides from the previous monitor run.
    Returns a mapping: {source_url: superseded_bool}.
    """
    monitor_path = paths.reports_dir / "zindi_monitor.json"
    if not monitor_path.exists():
        return {}

    try:
        payload = json.loads(monitor_path.read_text(encoding="utf-8"))
        old_flags = payload.get("compliance", {}).get("flags", [])
        mapping: dict[str, bool] = {}
        for f in old_flags:
            source = f.get("source")
            superseded = bool(f.get("superseded", False))
            if source:
                mapping[source] = superseded
        return mapping
    except Exception:
        return {}


def extract_compliance_flags(
    discussions: list,
    slug: str,
    headers: dict,
    superseded_map: dict[str, bool] | None = None,
) -> list:
    superseded_map = superseded_map or {}
    flagged = []
    for d in discussions:
        title    = d.get("title", "")
        pub_date = d.get("published_at", "")[:10]
        did      = d["id"]
        detail   = fetch_discussion_detail(slug, did, headers)
        body     = detail.get("body", "")
        comments = detail.get("comments", [])

        title_flagged   = is_compliance_relevant(title)
        body_flagged    = is_compliance_relevant(body)
        external_hits   = scan_external_data_mentions(title + " " + body)

        flagged_comments = []
        comment_external = []
        for c in comments:
            c_text = c.get("body", "")
            if is_compliance_relevant(c_text):
                flagged_comments.append({
                    "author": c.get("user", {}).get("username", "?"),
                    "text":   c_text[:300],
                    "date":   c.get("created_at", "")[:10],
                })
            comment_external += scan_external_data_mentions(c_text)

        all_external = list(set(external_hits + comment_external))

        if title_flagged or body_flagged or flagged_comments:
            # Build a concise flag text for downstream logic: prefer the title, else a preview
            flag_text = title if title_flagged else (body[:300] if body_flagged else "")
            source_url = f"https://zindi.africa/competitions/{slug}/discussions/{did}"
            flagged.append({
                "id":               did,
                "title":            title,
                "published":        pub_date,
                "body_preview":     body[:500],
                "flagged_comments": flagged_comments,
                "external_sources": all_external,
                "url": source_url,
                # New metadata for staleness tracking and provenance
                "flag":             flag_text,
                "source":           source_url,
                "scraped_at":       datetime.now(timezone.utc).isoformat(),
                "superseded":       superseded_map.get(source_url, False),
            })

    return flagged


# ── Section 3: Leaderboard ────────────────────────────────────────────────────

def fetch_leaderboard_intel(client: ZindiClient) -> dict:
    """Get current rank, top scores, and gap analysis."""
    import io, sys
    _buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = _buf
    my_rank = client._user.my_rank
    sys.stdout = _old

    return {
        "my_rank":    my_rank,
        "remaining":  client.remaining_submissions,
    }


# ── Section 4: Submission Board ───────────────────────────────────────────────

def fetch_submission_intel(client: ZindiClient) -> dict:
    """Pull all submissions with scores."""
    import io, sys
    _buf = io.StringIO()
    _old = sys.stdout
    sys.stdout = _buf
    subs: list[Any] = list(client._user.submission_board())
    sys.stdout = _old

    clean = []
    best_compliant = 0.0
    for raw in subs:
        if isinstance(raw, dict):
            s: dict[str, Any] = raw
        elif isinstance(raw, tuple):
            try:
                s = dict(raw)
            except Exception:
                s = {}
        else:
            s = {}

        score = s.get("public_score", 0.0) or 0.0
        clean.append({
            "id":      s["id"],
            "date":    s["created_at"][:10],
            "file":    s["filename"],
            "lb_f1":   score,
            "chosen":  s["chosen"],
            "comment": s["comment"],
        })
        if score > best_compliant:
            best_compliant = score

    chosen = [s for s in clean if s["chosen"]]
    return {
        "total":          len(clean),
        "best_score":     best_compliant,
        "chosen_count":   len(chosen),
        "chosen":         chosen,
        "all":            clean,
    }


# ── Report Writers ────────────────────────────────────────────────────────────

def write_compliance_log(
    flagged: list,
    all_discussions: list,
    comp_intel: dict,
    lb_intel: dict,
    sub_intel: dict,
    slug: str,
    paths,
) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Zindi Monitor Report",
        f"**Competition**: {slug}",
        f"**Generated**: {now}",
        "",
        "---",
        "",
        "## Competition Intelligence",
        "",
        f"- **Metric**            : {comp_intel.get('metric')}",
        f"- **Use probabilities** : {comp_intel.get('use_probabilities')}",
        f"- **External data**     : {'BANNED' if comp_intel.get('external_banned') else 'PERMITTED'}",
        f"- **AutoML**            : {'BANNED' if comp_intel.get('automl_banned') else 'PERMITTED'}",
        f"- **Code review**       : {comp_intel.get('code_review_tier')}",
        f"- **Daily limit**       : {comp_intel.get('daily_limit')}",
        f"- **Deadline**          : {comp_intel.get('end_time', '')[:10]}",
        f"- **Data files**        : {', '.join(comp_intel.get('datafiles', []))}",
        "",
        "---",
        "",
        "## Leaderboard Status",
        "",
        f"- **My rank**           : {lb_intel.get('my_rank')}",
        f"- **Remaining today**   : {lb_intel.get('remaining')}",
        f"- **Best LB score**     : {sub_intel.get('best_score')}",
        f"- **Chosen subs**       : {sub_intel.get('chosen_count')}/2",
        "",
        "---",
        "",
        "## Submission Board",
        "",
        f"{'ID':<12} {'Date':<12} {'LB F1':>12} {'Chosen':>8}  File",
        "-" * 100,
    ]
    for s in sub_intel.get("all", []):
        chosen = "YES" if s["chosen"] else "   "
        f1     = f"{s['lb_f1']:.9f}"
        lines.append(f"{s['id']:<12} {s['date']:<12} {f1:>12} {chosen:>8}  {s['file']}")

    lines += [
        "",
        "---",
        "",
        f"## Compliance Flags ({len(flagged)} flagged / {len(all_discussions)} total)",
        "",
    ]

    if not flagged:
        lines.append("No compliance issues detected.")
    else:
        for f in flagged:
            lines += [
                f"### {f['title']}",
                f"**Date**: {f['published']}",
                f"**URL**: {f['url']}",
                f"**External sources mentioned**: {f['external_sources'] or 'none'}",
                "",
                f"{f['body_preview']}",
                "",
            ]
            if f["flagged_comments"]:
                for c in f["flagged_comments"]:
                    lines.append(f"- **{c['author']}** ({c['date']}): {c['text']}")
            lines += ["", "---", ""]

    lines += [
        "",
        "## All Discussions",
        "",
        "| Date | Title | Comments |",
        "|------|-------|----------|",
    ]
    for d in all_discussions:
        title = d.get("title", "")[:60]
        date  = d.get("published_at", "")[:10]
        count = d.get("comments_count", 0)
        lines.append(f"| {date} | {title} | {count} |")

    (paths.reports_dir / "compliance_log.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"  ✅ compliance_log.md updated")


def write_monitor_json(
    comp_intel: dict,
    lb_intel: dict,
    sub_intel: dict,
    flagged: list,
    paths,
) -> None:
    """Machine-readable output for Skill 03 legality gate."""
    out = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "competition_intel": comp_intel,
        "leaderboard":       lb_intel,
        "submissions":       {
            "total":        sub_intel["total"],
            "best_score":   sub_intel["best_score"],
            "chosen_count": sub_intel["chosen_count"],
            "chosen":       sub_intel["chosen"],
        },
        "compliance": {
            "flagged_count":  len(flagged),
            "flagged_titles": [f["title"] for f in flagged],
            "flags": flagged,
            "external_sources_mentioned": list(set(
                src for f in flagged for src in f.get("external_sources", [])
            )),
            "agent_must_read": len(flagged) > 0,
        },
    }
    out_path = paths.reports_dir / "zindi_monitor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  ✅ zindi_monitor.json updated")


# ── State Update ──────────────────────────────────────────────────────────────

def update_state(
    comp_intel: dict,
    lb_intel: dict,
    sub_intel: dict,
    flagged: list,
    all_discussions: list,
    paths,
) -> None:
    store = SkillStateStore(paths.state_path)
    store.update(
        anchor_rank              = lb_intel.get("my_rank"),
        remaining_submissions    = lb_intel.get("remaining"),
        compliance               = {
            "last_checked":      datetime.now(timezone.utc).isoformat(),
            "total_discussions": len(all_discussions),
            "flagged_count":     len(flagged),
            "flagged_titles":    [f["title"] for f in flagged],
            "external_sources":  list(set(
                src for f in flagged for src in f.get("external_sources", [])
            )),
            "agent_must_read":   len(flagged) > 0,
        },
        last_updated             = datetime.now(timezone.utc).isoformat(),
    )
    print(f"  ✅ SKILL_STATE.json updated")


# ── Entry Point ───────────────────────────────────────────────────────────────

def run() -> dict:
    print(f"\n{'='*60}")
    print(f"SKILL 00 — Zindi Monitor")
    print(f"{'='*60}\n")

    paths  = resolve_competition_paths()
    config = ChallengeConfig.load()
    slug   = config.slug
    hdrs   = _get_headers()

    # ── 1. Competition page ───────────────────────────────────
    print("[1/4] Fetching competition intelligence...")
    try:
        comp_intel = fetch_competition_intel(slug, hdrs, config=config)
        print(f"  Source              : {comp_intel.get('intel_source')}")
        print(f"  Metric              : {comp_intel.get('metric')}")
        print(f"  Use probabilities   : {comp_intel.get('use_probabilities')}")
        print(f"  External data       : {'BANNED' if comp_intel.get('external_banned') else 'PERMITTED'}")
        print(f"  AutoML              : {'BANNED' if comp_intel.get('automl_banned') else 'PERMITTED'}")
        print(f"  Code review         : {comp_intel.get('code_review_tier')}")
        print(f"  Daily limit         : {comp_intel.get('daily_limit')}")
        print(f"  Total limit         : {comp_intel.get('total_limit')}")
        print(f"  Must select 2       : {comp_intel.get('must_select_2')}")
        print(f"  Seed required       : {comp_intel.get('seed_required')}")
        print(f"  Team size           : {comp_intel.get('team_size')}")
        print(f"  Public split        : {comp_intel.get('public_split_pct')}%")
        print(f"  Data files          : {len(comp_intel.get('datafiles', []))}")
        if comp_intel.get("external_hints"):
            print(f"  External file hints : {comp_intel['external_hints']}")
        if comp_intel.get("external_sources_on_data_page"):
            print(f"  External on data pg : {comp_intel['external_sources_on_data_page']}")
    except Exception as e:
        print(f"  ⚠️  Competition fetch failed: {e}")
        comp_intel = {}

    # ── 2. Discussions ────────────────────────────────────────
    print("\n[2/4] Scanning discussion board...")
    try:
        all_discussions = fetch_discussions(slug, hdrs)
        superseded_map  = load_superseded_flags(paths)
        flagged         = extract_compliance_flags(
            all_discussions, slug, hdrs, superseded_map=superseded_map
        )
        print(f"  Total discussions   : {len(all_discussions)}")
        print(f"  Compliance flags    : {len(flagged)}")
        if flagged:
            print(f"  Flagged titles:")
            for f in flagged:
                print(f"    🚨 {f['title']}")
                if f["external_sources"]:
                    print(f"       External sources: {f['external_sources']}")
    except Exception as e:
        print(f"  ⚠️  Discussion fetch failed: {e}")
        all_discussions, flagged = [], []

    # Derive an authoritative external_banned flag from explicit ban language in flags
    def _flag_unambiguous_ban(f: dict) -> bool:
        if f.get("superseded", False):
            return False
        txt = (f.get("flag") or "").lower()
        # Look for explicit ban/rule language only
        return any(k in txt for k in ("banned", "not allowed", "no external data", "external data is banned"))

    external_banned_from_flags = any(_flag_unambiguous_ban(f) for f in flagged)
    # If explicit bans are present in discussion flags, prefer that over scraped heuristics
    if external_banned_from_flags:
        comp_intel["external_banned"] = True
    else:
        # Leave scraped value as-is (False/True) when no explicit ban found in flags
        comp_intel.setdefault("external_banned", False)

    # ── 3. Leaderboard ────────────────────────────────────────
    print("\n[3/4] Fetching leaderboard status...")
    client: ZindiClient | None = None
    try:
        client = ZindiClient()
        client.select_competition(slug)
        lb_intel = fetch_leaderboard_intel(client)
        print(f"  My rank             : {lb_intel['my_rank']}")
        print(f"  Remaining today     : {lb_intel['remaining']}")
    except Exception as e:
        print(f"  ⚠️  Leaderboard fetch failed: {e}")
        lb_intel = {"my_rank": None, "remaining": None}

    # ── 4. Submission board ───────────────────────────────────
    print("\n[4/4] Fetching submission board...")
    try:
        if client is None:
            raise RuntimeError("Zindi client not available")
        sub_intel = fetch_submission_intel(client)
        print(f"  Total submissions   : {sub_intel['total']}")
        print(f"  Best LB score       : {sub_intel['best_score']:.9f}")
        print(f"  Chosen submissions  : {sub_intel['chosen_count']}/2")
        print(f"\n  {'ID':<12} {'Date':<12} {'LB F1':>12} {'Chosen':>8}  File")
        print(f"  {'-'*80}")
        for s in sub_intel["all"]:
            chosen = "YES" if s["chosen"] else "   "
            f1     = f"{s['lb_f1']:.9f}"
            print(f"  {s['id']:<12} {s['date']:<12} {f1:>12} {chosen:>8}  {s['file']}")
    except Exception as e:
        print(f"  ⚠️  Submission board failed: {e}")
        sub_intel = {"total": 0, "best_score": 0, "chosen_count": 0, "chosen": [], "all": []}

    # ── Write outputs ─────────────────────────────────────────
    print("\n[Writing outputs...]")
    write_compliance_log(flagged, all_discussions, comp_intel, lb_intel, sub_intel, slug, paths)
    write_monitor_json(comp_intel, lb_intel, sub_intel, flagged, paths)
    update_state(comp_intel, lb_intel, sub_intel, flagged, all_discussions, paths)

    # ── Summary ───────────────────────────────────────────────
    status = "WARNING" if flagged else "CLEAR"
    print(f"\n{'='*60}")
    print(f"MONITOR STATUS: {status}")
    print(f"{'='*60}")
    print(f"Rank             : {lb_intel.get('my_rank')}")
    print(f"Best LB score    : {sub_intel.get('best_score'):.9f}")
    print(f"Compliance flags : {len(flagged)}")
    print(f"Remaining today  : {lb_intel.get('remaining')}")

    if flagged:
        print(f"\n⚠️  Read reports/compliance_log.md before proceeding to Skill 03")
    else:
        print(f"\n✅ No compliance issues — safe to proceed to Skill 03")

    return {
        "status":         status,
        "my_rank":        lb_intel.get("my_rank"),
        "best_lb_score":  sub_intel.get("best_score"),
        "flagged_count":  len(flagged),
        "remaining":      lb_intel.get("remaining"),
        "metric":         comp_intel.get("metric"),
        "external_banned": comp_intel.get("external_banned"),
        "external_sources_in_discussions": list(set(
            src for f in flagged for src in f.get("external_sources", [])
        )),
    }


if __name__ == "__main__":
    result = run()
    print(f"\n{json.dumps({k: v for k, v in result.items()}, indent=2)}")
