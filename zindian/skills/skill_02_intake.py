"""
Skill 02 — Challenge Intake
Reads the competition API and populates challenge_config.json
with all rules, limits, and constraints extracted from the response.
Must run after Skill 00 (compliance check).
"""

import tabula.skill_state_autopatch  # noqa
import json
import requests
import tempfile
from datetime import datetime, timezone
import difflib

from zindian.paths import CompetitionPaths, resolve_competition_paths
from zindian.config import ConfigNotPopulated
from zindian.state import SkillStateStore
from zindian.schemas import validate_challenge_config

BASE_URL = "https://api.zindi.africa/v1/competitions"


def fetch_competition(slug: str, headers: dict) -> dict:
    url = f"{BASE_URL}/{slug}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json().get("data", {})


def extract_config(data: dict, slug: str) -> dict:
    """
    Extract all rule fields from raw competition API response.
    These values are ground truth — hardcoding nothing.
    """
    sections = data.get("sections", [])

    # Extract rules text from sections
    rules_text = ""
    for s in sections:
        ct = s.get("content_text", "")
        if ct:
            rules_text += ct + " "

    # Parse known fields from the API response; do not hardcode defaults.
    # These fields should be treated as authoritative when provided by the API.
    name = data.get("name") or data.get("title") or f"{slug}"
    subtitle = data.get("subtitle")
    end_time = data.get("end_time")

    # Metric must come from API or be parsed from sections. If missing, write null.
    metric = data.get("metric")
    if metric is None and rules_text:
        text_lower = rules_text.lower()
        import re

        metric_patterns = [
            r"evaluation metric.*?is\s+([a-z0-9_ ]+score|rmse|mae|auc|accuracy|log.?loss)",
            r"scored using\s+([a-z0-9_ ]+score|rmse|mae|auc|accuracy|log.?loss)",
            r"error metric.*?(f1|rmse|mae|auc|accuracy|log.?loss|f-1)",
        ]
        for pattern in metric_patterns:
            _match = re.search(pattern, text_lower)
            if _match:
                raw = _match.group(1).strip()
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

        if not metric:
            if "f1 score" in text_lower or "f-1 score" in text_lower:
                metric = "f1_score"
            elif "root mean squared" in text_lower or "rmse" in text_lower:
                metric = "rmse"
            elif "mean absolute" in text_lower or " mae" in text_lower:
                metric = "mae"
            elif "log loss" in text_lower or "logloss" in text_lower:
                metric = "log_loss"
            elif "area under" in text_lower or " auc " in text_lower:
                metric = "auc"
            elif "accuracy" in text_lower:
                metric = "accuracy"

    metric_direction = data.get("metric_direction")

    # Determine task_type dynamically if possible
    task_type = data.get("task_type") or data.get("type")
    if task_type is None and metric is not None:
        m = str(metric).lower()
        if m in ("rmse", "mae", "mse", "mape", "rmsle"):
            task_type = "regression"
        elif m in (
            "auc",
            "f1_score",
            "f1",
            "accuracy",
            "log_loss",
            "logloss",
            "cross_entropy",
        ):
            task_type = "classification"
        else:
            task_type = None

    # Derive use_probabilities from metric/rules when possible.
    use_probabilities = data.get("use_probabilities")
    if use_probabilities is None:
        if rules_text:
            text_lower = rules_text.lower()
            if (
                "do not set thresholds" in text_lower
                or "raw probabilities" in text_lower
                or "if the error metric requires probabilities" in text_lower
            ):
                use_probabilities = True
            elif "hard labels" in text_lower or "0/1 labels" in text_lower:
                use_probabilities = False

        if use_probabilities is None and metric is not None:
            m = str(metric).lower()
            if m in ("log_loss", "logloss", "cross_entropy", "auc"):
                use_probabilities = True
            elif m in ("f1_score", "f1", "accuracy", "rmse", "mae"):
                use_probabilities = False
            else:
                use_probabilities = None

    # Limits and splits: take from API when provided, otherwise leave null.
    daily_limit = data.get("daily_limit")
    total_limit = data.get("total_limit")
    public_split = data.get("public_split_pct") or data.get("public_split")
    private_split = data.get("private_split_pct") or data.get("private_split")

    # Team information from the API; do NOT hardcode EY-specific overrides here.
    team_allowed = data.get("team_allowed")
    max_team_size = data.get("max_team_size") or data.get("team_size")

    code_review_tier = data.get("code_review_tier")
    code_review_hours = data.get("code_review_hours")

    allowed_external_data = data.get("allowed_external_data")
    automl_permitted = data.get("automl_permitted")
    data_modality = data.get("data_modality") or data.get("modality")
    domain = data.get("domain")

    skills_required = data.get("skills", [])
    banned_features = data.get("banned_features", [])

    target_col = data.get("target_col") or data.get("target_column")
    target_domain_bounds = data.get("target_domain_bounds") or {
        "min": None,
        "max": None,
    }
    reproducibility = data.get("reproducibility") or {"seed": 42}

    config = {
        "name": name,
        "slug": slug,
        "subtitle": subtitle,
        "end_time": end_time,
        "task_type": task_type,
        "target_col": target_col,
        "target_domain_bounds": target_domain_bounds,
        "metric": metric,
        "metric_direction": metric_direction,
        "submission_format": data.get("submission_format"),
        "use_probabilities": use_probabilities,
        "daily_limit": daily_limit,
        "total_limit": total_limit,
        "public_split_pct": public_split,
        "private_split_pct": private_split,
        "team_allowed": team_allowed,
        "max_team_size": max_team_size,
        "code_review_tier": code_review_tier,
        "code_review_hours": code_review_hours,
        "allowed_external_data": allowed_external_data,
        "automl_permitted": automl_permitted,
        "data_modality": data_modality,
        "domain": domain,
        "skills_required": skills_required,
        "banned_features": banned_features,
        "compliance_notes": [],
        "reproducibility": reproducibility,
        "populated_at": datetime.now(timezone.utc).isoformat(),
    }

    # If metric is missing, attempt to fallback to monitor output.
    field_sources = {}
    for k in config:
        field_sources[k] = "api" if config[k] is not None else None

    if config["metric"] is None:
        try:
            paths = resolve_competition_paths(slug=slug)
            monitor_path = paths.reports_dir / "zindi_monitor.json"
            if monitor_path.exists():
                with open(monitor_path, encoding="utf-8") as f:
                    mon = json.load(f)
                ci = mon.get("competition_intel", {})
                # Map fallback fields
                if config.get("metric") is None and ci.get("metric") is not None:
                    config["metric"] = ci.get("metric")
                    field_sources["metric"] = "monitor"
                if (
                    config.get("use_probabilities") is None
                    and ci.get("use_probabilities") is not None
                ):
                    config["use_probabilities"] = ci.get("use_probabilities")
                    field_sources["use_probabilities"] = "monitor"
                # external_banned -> allowed_external_data (invert)
                if (
                    config.get("allowed_external_data") is None
                    and ci.get("external_banned") is not None
                ):
                    config["allowed_external_data"] = not bool(
                        ci.get("external_banned")
                    )
                    field_sources["allowed_external_data"] = "monitor"
                # automl_banned -> automl_permitted (invert)
                if (
                    config.get("automl_permitted") is None
                    and ci.get("automl_banned") is not None
                ):
                    config["automl_permitted"] = not bool(ci.get("automl_banned"))
                    field_sources["automl_permitted"] = "monitor"
                if (
                    config.get("daily_limit") is None
                    and ci.get("daily_limit") is not None
                ):
                    config["daily_limit"] = ci.get("daily_limit")
                    field_sources["daily_limit"] = "monitor"
                if (
                    config.get("total_limit") is None
                    and ci.get("total_limit") is not None
                ):
                    config["total_limit"] = ci.get("total_limit")
                    field_sources["total_limit"] = "monitor"
                if (
                    config.get("code_review_tier") is None
                    and ci.get("code_review_tier") is not None
                ):
                    config["code_review_tier"] = ci.get("code_review_tier")
                    field_sources["code_review_tier"] = "monitor"
                if (
                    config.get("max_team_size") is None
                    and ci.get("team_size") is not None
                ):
                    config["max_team_size"] = ci.get("team_size")
                    field_sources["max_team_size"] = "monitor"
                if (
                    config.get("public_split_pct") is None
                    and ci.get("public_split_pct") is not None
                ):
                    config["public_split_pct"] = ci.get("public_split_pct")
                    field_sources["public_split_pct"] = "monitor"

                # Log which fields came from API vs monitor fallback
                api_fields = [k for k, v in field_sources.items() if v == "api"]
                monitor_fields = [k for k, v in field_sources.items() if v == "monitor"]
                if monitor_fields:
                    print("\n--- Fallback Applied From zindi_monitor.json ---")
                    print(
                        "Fields from API: ",
                        ", ".join(api_fields) if api_fields else "(none)",
                    )
                    print("Fields from monitor fallback: ", ", ".join(monitor_fields))
        except Exception as e:
            # Non-fatal; we'll handle missing metric later
            print(f"Warning: monitor fallback failed: {e}")

    # Derive task_type and use_probabilities after fallback if still None
    if config.get("task_type") is None and config.get("metric") is not None:
        m = str(config.get("metric")).lower()
        if m in ("rmse", "mae", "mse", "mape", "rmsle"):
            config["task_type"] = "regression"
        elif m in (
            "auc",
            "f1_score",
            "f1",
            "accuracy",
            "log_loss",
            "logloss",
            "cross_entropy",
        ):
            config["task_type"] = "classification"

    if config.get("use_probabilities") is None and config.get("metric") is not None:
        m = str(config.get("metric")).lower()
        if any(
            x in m
            for x in ("log_loss", "logloss", "cross_entropy", "auc", "area_under_curve")
        ):
            config["use_probabilities"] = True
        elif any(
            x in m
            for x in (
                "f1",
                "accuracy",
                "rmse",
                "mae",
                "root_mean_squared",
                "mean_absolute",
                "bleu",
                "rmsle",
                "mse",
                "mape",
                "r2",
                "r_squared",
            )
        ):
            config["use_probabilities"] = False

    # Derive metric_direction from metric when not provided
    if config.get("metric_direction") is None and config.get("metric") is not None:
        m = str(config.get("metric")).lower()
        if any(
            x in m
            for x in (
                "f1",
                "accuracy",
                "auc",
                "area_under_curve",
                "bleu",
                "straight_accuracy",
                "r2",
                "r_squared",
                "score",
            )
        ):
            config["metric_direction"] = "maximize"
        elif any(
            x in m
            for x in (
                "rmse",
                "mae",
                "log_loss",
                "logloss",
                "cross_entropy",
                "root_mean_squared",
                "mean_absolute",
                "rmsle",
                "mse",
                "mape",
                "loss",
            )
        ):
            config["metric_direction"] = "minimize"
        else:
            config["metric_direction"] = None

    # Build final compliance notes from resolved values
    up = config.get("use_probabilities")
    cn = []

    # Check if multi-target before adding single-target compliance notes
    target_config = config.get("target_config")
    is_multi_target = (
        target_config is not None and len(target_config.get("targets", [])) > 1
    )

    if is_multi_target:
        # Multi-target: describe each target's requirements
        cn.append("Multi-target competition - see target_config for per-target metrics")
    elif up is True:
        cn.append(
            "use_probabilities=True: submit raw float probabilities, do NOT threshold"
        )
    elif up is False:
        if config.get("task_type") == "regression":
            cn.append("Regression task: submit continuous numeric predictions")
        else:
            cn.append("use_probabilities=False: submit hard 0/1 integer labels only")
    else:
        cn.append("use_probabilities: unknown — confirm from competition page")
    if config.get("allowed_external_data") is False:
        cn.append("No external data allowed — provided datasets only")
    if config.get("banned_features"):
        cn.append("Banned features: " + ", ".join(config.get("banned_features") or []))
    if config.get("code_review_tier"):
        cn.append(
            f"Code review tier: {config.get('code_review_tier')} ({config.get('code_review_hours') or 'hours unspecified'})"
        )
    cn.extend(
        [
            "Final 5 days: no new team members",
            "Must select 2 submissions before deadline",
            "Always set random seed for reproducibility",
            "Open source packages only — no paid services",
        ]
    )
    config["compliance_notes"] = cn

    # -- Structured CV limitations (queryable by skill_11, skill_17, three_lens) --
    # Detect whether the competition requires temporal holdout evaluation.
    # This cannot be inferred from data alone — it must come from either a
    # structured config field ("temporal_holdout") set by the operator, or
    # from the compliance notes text.  Never infer it from feature column names,
    # which would produce false positives on competitions with monthly features
    # but a random train/test split.
    temporal_holdout_required: bool = bool(config.get("temporal_holdout", False))
    if not temporal_holdout_required:
        for note in cn:
            if "held-out time period" in note.lower() or "temporal generalization" in note.lower():
                temporal_holdout_required = True
                break

    # Detect whether a row-level temporal column exists for TimeSeriesSplit.
    # This is set by the operator via "temporal_col" in config; skill_04 does
    # NOT write this because column-name pattern detection is not reliable.
    temporal_cv_feasible: bool = bool(config.get("temporal_col"))

    if temporal_holdout_required:
        reason = (
            "No row-level date/period column in training data — temporal structure "
            "exists only in feature column name suffixes, not as a splittable row "
            "attribute."
            if not temporal_cv_feasible
            else "Row-level temporal column present; TimeSeriesSplit feasible."
        )
        config["cv_limitations"] = {
            "temporal_holdout_required": True,
            "temporal_cv_feasible": temporal_cv_feasible,
            "reason": reason,
            "fallback_strategy": config.get("cv_strategy", {}).get("type", "stratified"),
            "known_risk": (
                "OOF scores likely optimistic relative to true held-out time period "
                "performance — local CV cannot replicate the actual evaluation split."
                if not temporal_cv_feasible
                else None
            ),
        }
    else:
        config.setdefault("cv_limitations", None)

    return config


def write_config(config: dict, paths: CompetitionPaths) -> None:
    import shutil

    path = paths.config_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp_path = tmp.name
    shutil.move(tmp_path, path)
    print(f"[OK] challenge config populated -> {path}")


def update_skill_state(slug: str, paths: CompetitionPaths) -> None:
    state_path = paths.state_path
    if not state_path.exists():
        return
    store = SkillStateStore(state_path)
    # Per SoT: only update dag phase if not beyond Phase 1
    state = store.read()
    current_phase = state.get("dag_phase")
    if current_phase in (None, "uninitialized", "phase_0_foundation"):
        store.update(dag_phase="phase_1_integrity")
        print(f"[OK] {state_path} -> dag_phase: phase_1_integrity")
    else:
        print(f"[INFO]  SKIP dag_phase update (current phase: {current_phase})")


def _detect_multi_target_from_submission(sample_submission_path, config):
    """Detect if competition has multiple targets from submission format.

    Returns target_config dict if multi-target, None otherwise.
    A multi-column submission (e.g. TargetF1 + TargetRAUC) for a single
    target is NOT multi-target — it's a dual-column classification format.
    """
    import pandas as pd

    # If config already has a target_config with submission_columns, skip detection
    existing_tc = config.get("target_config") or {}
    if existing_tc.get("submission_columns"):
        print("  [INFO] submission_columns explicit — skipping multi-target detection")
        return None

    df = pd.read_csv(sample_submission_path)
    cols = [c for c in df.columns if c.lower() not in ("id", "team_id", "uniqueid")]

    if len(cols) <= 1:
        return None  # Single-target

    # Check if this is a single-target dual-column format (binary + probability)
    # Pattern: 2 columns where one ends in "F1" and the other in "RAUC" or "Prob"
    if len(cols) == 2:
        col_set = {c.lower() for c in cols}
        is_dual_format = any(
            c.endswith("f1") or c.endswith("_f1") for c in col_set
        ) and any("rauc" in c or "prob" in c or "auc" in c for c in col_set)
        if is_dual_format:
            print(
                f"  [INFO] Dual-column classification format ({', '.join(cols)})"
                " — single target with binary + probability output"
            )
            return None  # Not multi-target

    # Multi-target detected
    targets = []
    for col in cols:
        # Infer task_type from column values
        sample_vals = df[col].dropna().head(100)
        # Convert to numeric for comparison
        try:
            numeric_vals = pd.to_numeric(sample_vals, errors="coerce").dropna()
            is_binary = set(numeric_vals.unique()).issubset({0, 1, 0.0, 1.0})
            is_prob = (numeric_vals >= 0).all() and (numeric_vals <= 1).all()
        except Exception:
            is_binary = False
            is_prob = False

        task_type = "classification" if (is_binary or is_prob) else "regression"

        targets.append(
            {
                "name": col,
                "task_type": task_type,
                "metric": "rmse" if task_type == "regression" else "f1",
                "metric_direction": (
                    "minimize" if task_type == "regression" else "maximize"
                ),
                "weight": 1.0 / len(cols),
                "target_domain_bounds": (
                    None
                    if task_type == "classification"
                    else {"min": None, "max": None}
                ),
            }
        )

    # A12: mixed-task requires recombination policy
    has_classification = any(t["task_type"] == "classification" for t in targets)
    has_regression = any(t["task_type"] == "regression" for t in targets)

    target_config = {
        "targets": targets,
        "composite_direction": "minimize_composite_distance",
    }

    if has_classification and has_regression:
        target_config["pseudo_label_recombination_policy"] = (
            "freeze_unaugmented_targets_at_original"
        )

    return target_config


def run(
    slug: str | None = None,
    headers: dict | None = None,
    dry_run: bool = False,
    merge: bool = False,
) -> dict:
    if slug is None:
        import os

        slug = os.environ.get("COMPETITION_SLUG")
        if not slug:
            try:
                from zindian.config import ChallengeConfig

                slug = ChallengeConfig.load().slug
            except Exception:
                pass
        if not slug:
            raise ValueError(
                "slug must be provided or set via COMPETITION_SLUG environment variable"
            )

    # Lazy import headers only if needed
    if headers is None:
        try:
            from zindian.zindi_client import ZindiClient

            client = ZindiClient()
            headers = client._headers
        except Exception as e:
            print(f"[WARN] Failed to load Zindi Client for headers: {e}")
            # Network isolation - will use monitor fallback
            headers = {"Accept": "application/json"}

    paths = resolve_competition_paths(slug=slug)

    print(f"\n{'=' * 60}")
    print("SKILL 02 — Challenge Intake")
    print(f"Competition: {slug}")
    print(f"{'=' * 60}\n")

    # Try API first, fall back to monitor file if network isolated
    try:
        print("Fetching competition details from API...")
        data = fetch_competition(slug, headers)
    except Exception as e:
        print(f"[WARN]  API fetch failed: {e}")
        print("Attempting fallback to zindi_monitor.json...")
        monitor_path = paths.reports_dir / "zindi_monitor.json"
        if monitor_path.exists():
            with open(monitor_path, encoding="utf-8") as f:
                mon = json.load(f)
            data = mon.get("competition_intel", {})
            print("[OK] Using competition intel from zindi_monitor.json")
        else:
            raise RuntimeError(
                f"API unavailable and no zindi_monitor.json found at {monitor_path}"
            )

    print("Extracting config fields...")
    config = extract_config(data, slug)

    existing = {}
    if paths.config_path.exists():
        try:
            existing = json.loads(paths.config_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # If merge requested, combine with existing challenge_config.json without overwriting non-null fields
    if merge:
        merged = dict(existing)  # start from existing; preserve extra fields
        for k, v in config.items():
            # Only set if existing value is None or missing
            if merged.get(k) is None:
                merged[k] = v
        final_to_write = merged
    else:
        # Prefer new scraped config, but fall back to existing config for any fields that are None or missing
        merged = dict(config)
        for k in list(merged.keys()):
            if merged[k] is None and existing.get(k) is not None:
                merged[k] = existing[k]
        # Also preserve any fields that are in existing but not in config
        for k, v in existing.items():
            if k not in merged:
                merged[k] = v
        final_to_write = merged

    # Validate before any mutation of challenge_config.json or SKILL_STATE.json.
    if final_to_write.get("metric") is None:
        raise ConfigNotPopulated(
            "Required field 'metric' is null after intake from API and fallback."
        )
    if final_to_write.get("use_probabilities") is None:
        raise ConfigNotPopulated(
            "Derived field 'use_probabilities' is null — cannot infer from metric."
        )

    # Ensure allowed_external_data and automl_permitted default to False if still None/missing
    if final_to_write.get("allowed_external_data") is None:
        final_to_write["allowed_external_data"] = False
    if final_to_write.get("automl_permitted") is None:
        final_to_write["automl_permitted"] = False

    # Multi-target detection (A11)
    if not dry_run:
        try:
            sample_sub_path = paths.data_raw_dir / "SampleSubmission.csv"
            if sample_sub_path.exists():
                target_config = _detect_multi_target_from_submission(
                    sample_sub_path, final_to_write
                )
                if target_config:
                    final_to_write["target_config"] = target_config
                    print(
                        f"[OK] Multi-target detected: {len(target_config['targets'])} targets"
                    )
        except Exception as e:
            print(f"Multi-target detection skipped: {e}")

    if dry_run:
        print("\n--- DRY RUN: challenge_config.json that WOULD be written ---\n")
        print(json.dumps(final_to_write, indent=2))
    else:
        # Only write config during allowed intake phases per SoT
        store = SkillStateStore(paths.state_path)
        current_phase = store.read().get("dag_phase")
        # Allow write in INIT mode (phase_1_complete from skill_01 is still INIT)
        allowed_write_phases = (
            None,
            "uninitialized",
            "phase_0_foundation",
            "phase_1",
            "phase_1_complete",
            "phase_1_integrity",
            "phase_1_integrity_locked",  # Bootstrap phase string
        )
        if current_phase in allowed_write_phases:
            write_config(final_to_write, paths)
            # Operator-agreed validation before advancing the DAG phase
            try:
                validate_challenge_config(final_to_write)
                if final_to_write.get("task_type") and final_to_write.get("target_col"):
                    update_skill_state(slug, paths)
                else:
                    print(
                        "[INFO]  Skipping dag_phase update because task_type or target_col is still null/unconfigured."
                    )
            except Exception as e:
                print(
                    f"[INFO]  Skipping dag_phase update because config is not fully validated: {e}"
                )
        else:
            print(
                f"[WARN]  Skipping challenge_config.json write — current phase '{current_phase}' prohibits config mutation."
            )

    # If merge and dry_run, show a concise diff between existing and final_to_write
    if merge and dry_run:
        existing_text = ""
        if paths.config_path.exists():
            existing_text = paths.config_path.read_text(encoding="utf-8")
        new_text = json.dumps(final_to_write, indent=2)
        diff = difflib.unified_diff(
            existing_text.splitlines(),
            new_text.splitlines(),
            fromfile=str(paths.config_path),
            tofile="(merged)",
            lineterm="",
        )
        print("\n--- DIFF (existing -> merged) ---")
        for line in diff:
            print(line)

    print("\n--- Config Summary ---")
    print(f"Name       : {final_to_write.get('name')}")
    print(
        f"Metric     : {final_to_write.get('metric')} ({final_to_write.get('metric_direction')})"
    )
    print(f"Modality   : {final_to_write.get('data_modality')}")
    print(f"Daily limit: {final_to_write.get('daily_limit')}")
    print(f"Total limit: {final_to_write.get('total_limit')}")
    print(f"Public split: {final_to_write.get('public_split_pct')}%")
    print(f"Private split: {final_to_write.get('private_split_pct')}%")
    print(f"Use probabilities: {final_to_write.get('use_probabilities')}")
    print(f"External data: {final_to_write.get('allowed_external_data')}")
    print(f"Banned features: {final_to_write.get('banned_features')}")
    print("\nCompliance notes:")
    for note in final_to_write.get("compliance_notes", []):
        print(f"  [WARN]  {note}")

    # R5: Add infrastructure block if missing
    if "infrastructure" not in final_to_write:
        final_to_write["infrastructure"] = {
            "hardware_type": "cpu",
            "region": "us-east-1",
            "tdp_watts": 15.0,
            "pue": 1.0,
            "carbon_intensity_gco2_per_kwh": 494.0
        }
        print("\n[R5] Added infrastructure block for carbon tracking")

    return {"status": "OK", "config": final_to_write}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Skill 02 — Challenge Intake")
    parser.add_argument("slug", help="competition slug e.g. ey-frogs")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Print config without writing it",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        dest="merge",
        help="Merge with existing challenge_config.json without overwriting non-null fields",
    )
    args = parser.parse_args()

    # Minimal headers placeholder — real use should provide auth in environment
    headers = {"Accept": "application/json"}
    try:
        run(args.slug, headers, dry_run=args.dry_run, merge=args.merge)
    except ConfigNotPopulated as e:
        print(f"ERROR: {e}")
        raise
