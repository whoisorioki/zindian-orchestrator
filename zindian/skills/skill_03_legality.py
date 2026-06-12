"""
Skill 03 — Legality Gate

Two functions:
  - synthesise_feature_policy(monitor_data, config)
  - check_planned_features(policy, planned_features)

Entry point `run(slug, planned_features=None)` runs both steps, writes
`reports/feature_policy.json` and `reports/legality_report.md`, and updates
`SKILL_STATE.json` only when not downgrading the DAG phase.

Generalisable: reads from `zindi_monitor.json`, `challenge_config.json`, and
`SKILL_STATE.json`. Does not hardcode specific data sources.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional

from zindian.paths import resolve_competition_paths
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore


def _normalize_policy_token(value: Any) -> str:
    token = str(value).strip().lower()
    return "_".join(token.replace("-", "_").split())


def _collect_banned_features(
    monitor_data: Mapping[str, Any], config: Mapping[str, Any]
) -> List[str]:
    banned: List[str] = []

    monitor_root = (
        monitor_data.get("banned_features", [])
        if isinstance(monitor_data, Mapping)
        else []
    )
    if isinstance(monitor_root, list):
        banned.extend(monitor_root)
    elif monitor_root:
        banned.append(monitor_root)

    competition_intel = (
        monitor_data.get("competition_intel", {})
        if isinstance(monitor_data, Mapping)
        else {}
    )
    nested_monitor = (
        competition_intel.get("banned_features", [])
        if isinstance(competition_intel, Mapping)
        else []
    )
    if isinstance(nested_monitor, list):
        banned.extend(nested_monitor)
    elif nested_monitor:
        banned.append(nested_monitor)

    config_banned = config.get("banned_features", []) if config else []
    if isinstance(config_banned, list):
        banned.extend(config_banned)
    elif config_banned:
        banned.append(config_banned)

    return list(dict.fromkeys(str(item) for item in banned if item is not None))


def _normalize_planned_feature_entries(entries: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []

    def add_entry(entry: Any) -> None:
        if entry is None:
            return
        if isinstance(entry, (list, tuple, set)):
            for item in entry:
                add_entry(item)
            return
        if isinstance(entry, str):
            normalized.append({"name": entry, "transforms": [], "uses_lat_lon": False})
            return
        if isinstance(entry, Mapping):
            normalized.append(
                {
                    "name": entry.get("name") or entry.get("feature") or str(entry),
                    "source": entry.get("source", ""),
                    "transforms": entry.get("transforms", []),
                    "uses_lat_lon": entry.get("uses_lat_lon", False),
                }
            )
            return
        normalized.append({"name": str(entry), "transforms": [], "uses_lat_lon": False})

    add_entry(entries)
    return normalized


def synthesise_feature_policy(
    monitor_data: Dict[str, Any], config: Mapping[str, Any], flagged_titles: List[str]
) -> Dict[str, Any]:
    """Derive a generic feature policy from monitor output and config.

    The policy is intentionally generic and avoids naming specific datasets.
    """
    comp = (
        monitor_data.get("competition_intel", {})
        if isinstance(monitor_data, dict)
        else {}
    )

    # Allowed external sources (from data page hints) — keep as advisory
    allowed_data_sources = comp.get(
        "allowed_data_sources", ["competition_provided_only"]
    )

    # Banned transformations and features: combine monitor and config
    banned_transformations = _collect_banned_features(monitor_data, config)

    # Lat/Lon permitted if no spatial bans mention 'derived_spatial' or 'spatial'
    lat_lon_permitted = True
    low = [_normalize_policy_token(b) for b in banned_transformations]
    if any("spatial" in s for s in low) or any(
        "latitude" in s or "longitude" in s for s in low
    ):
        lat_lon_permitted = False

    external_data_permitted = not comp.get("external_banned", True)
    automl_permitted = not comp.get("automl_banned", False)
    use_probabilities = (
        comp.get("use_probabilities")
        if comp.get("use_probabilities") is not None
        else config.get("use_probabilities")
    )
    metric = comp.get("metric") or config.get("metric")

    policy = {
        "allowed_data_sources": allowed_data_sources,
        "allowed_sources": allowed_data_sources,
        "banned_transformations": banned_transformations,
        "lat_lon_permitted_as_feature": lat_lon_permitted,
        "coordinate_features_permitted": lat_lon_permitted,
        "external_data_permitted": external_data_permitted,
        "automl_permitted": automl_permitted,
        "use_probabilities": use_probabilities,
        "metric": metric,
        "output_format": "probabilities" if use_probabilities else "hard_labels_0_1",
        "synthesised_at": datetime.now(timezone.utc).isoformat(),
        "source_flags": flagged_titles,
    }

    return policy


def check_planned_features(
    policy: Dict[str, Any], planned_features: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Check each planned feature against the policy.

    planned_features: list of dicts with keys: `name`, `source` (optional),
    `transforms` (list), `uses_lat_lon` (bool).
    Returns list of results: {name, status: PASS|WARN|BLOCK, reason, blocks}
    """
    results: List[Dict[str, Any]] = []

    allowed_sources = {
        _normalize_policy_token(source)
        for source in policy.get(
            "allowed_data_sources", policy.get("allowed_sources", [])
        )
    }
    external_ok = policy.get("external_data_permitted", True)
    banned_trans = {
        _normalize_policy_token(transform)
        for transform in policy.get("banned_transformations", [])
    }
    lat_ok = policy.get(
        "coordinate_features_permitted",
        policy.get("lat_lon_permitted_as_feature", True),
    )

    for f in planned_features:
        name = f.get("name")
        source = _normalize_policy_token(f.get("source") or "")
        transforms = [_normalize_policy_token(t) for t in f.get("transforms", [])]
        uses_lat_lon = bool(f.get("uses_lat_lon", False))

        status = "PASS"
        reason = ""
        blocks = False

        # External source check
        if source and not external_ok and source not in allowed_sources:
            status = "BLOCK"
            reason = f"External data source '{source}' is not permitted by policy"
            blocks = True

        # Transformation check
        if not blocks:
            offending = [t for t in transforms if t in banned_trans]
            if offending:
                status = "BLOCK"
                reason = f"Banned transformation(s) detected: {offending}"
                blocks = True

        # Lat/Lon usage
        if not blocks and uses_lat_lon and not lat_ok:
            status = "BLOCK"
            reason = "Use of latitude/longitude as features is prohibited by policy"
            blocks = True

        # Warnings: using unknown sources when external allowed but not listed
        if (
            not blocks
            and source
            and external_ok
            and allowed_sources
            and source not in allowed_sources
        ):
            status = "WARN"
            reason = (
                f"Source '{source}' not explicitly listed in allowed sources (advisory)"
            )

        results.append(
            {
                "name": name,
                "source": source,
                "transforms": transforms,
                "uses_lat_lon": uses_lat_lon,
                "status": status,
                "reason": reason,
                "blocks": blocks,
            }
        )

    return results


def _write_feature_policy(paths, policy: Dict[str, Any]) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    p = paths.reports_dir / "feature_policy.json"
    p.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"  ✅ feature_policy.json written -> {p}")


def _write_legality_report(
    paths, checks: List[Dict[str, Any]], policy: Dict[str, Any]
) -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    p = paths.reports_dir / "legality_report.md"
    lines = [
        "# Legality Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Policy",
        "",
        json.dumps(policy, indent=2),
        "",
        "## Feature Checks",
        "",
    ]
    for c in checks:
        lines += [
            f"- **{c['name']}**: {c['status']}  ",
            f"  - reason: {c['reason']}",
            f"  - blocks: {c['blocks']}",
            "",
        ]

    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✅ legality_report.md written -> {p}")


def run(
    slug: Optional[str] = None, planned_features: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}")
    print("SKILL 03 — Legality Gate")
    print(f"{'=' * 60}\n")

    paths = resolve_competition_paths(slug=slug)
    # Load monitor output if available
    monitor_path = paths.reports_dir / "zindi_monitor.json"
    monitor_data = {}
    flagged_titles = []
    if monitor_path.exists():
        monitor_data = json.loads(monitor_path.read_text(encoding="utf-8"))
        flagged_titles = monitor_data.get("compliance", {}).get("flagged_titles", [])
        print(f"  Loaded monitor data from {monitor_path}")
    else:
        print(
            f"  ⚠️  zindi_monitor.json not found at {monitor_path} — proceeding with config only"
        )

    # Load config
    try:
        cfg = ChallengeConfig.load()
        print(f"  Loaded challenge_config for {cfg.slug}")
    except Exception:
        cfg = None
        print("  ⚠️  challenge_config.json not available or invalid")

    # Synthesise feature policy
    policy = synthesise_feature_policy(
        monitor_data, cfg._data if cfg is not None else {}, flagged_titles
    )
    _write_feature_policy(paths, policy)

    # Decide planned_features source
    if planned_features is None:
        store = SkillStateStore(paths.state_path)
        state = store.read()
        # Try to read explicit planned_features from state
        planned_features = state.get("planned_features")
        if not planned_features:
            # Fallback: use anchor feature summary if available
            af = state.get("anchor_features")
            planned_features = af

    # Normalize planned_features into the expected structure
    normalized = _normalize_planned_feature_entries(planned_features)

    checks = check_planned_features(policy, normalized)
    _write_legality_report(paths, checks, policy)

    # Determine overall status
    blocked_reasons = [c for c in checks if c.get("blocks")]
    status = "GO" if not blocked_reasons else "BLOCKED"

    # Update SKILL_STATE.json but do not downgrade dag_phase
    store = SkillStateStore(paths.state_path)
    state = store.read()
    patch = {
        "legality_status": status,
        "feature_policy_written": str(paths.reports_dir / "feature_policy.json"),
        "last_legality_checked": datetime.now(timezone.utc).isoformat(),
    }
    # Only advance dag_phase if current is before phase_2_legality_checked
    current_phase = state.get("dag_phase")
    if status == "GO" and (
        current_phase in (None, "uninitialized", "phase_0_foundation")
        or (isinstance(current_phase, str) and current_phase.startswith("phase_1_"))
    ):
        patch["dag_phase"] = "phase_2_legality_checked"

    store.update(**patch)
    print(f"  ✅ SKILL_STATE.json updated with legality_status={status}")

    result = {
        "status": status,
        "blocked_reasons": [c for c in checks if c.get("blocks")],
        "checks": checks,
        "policy": policy,
    }

    return result


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
