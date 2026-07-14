"""Skill Orchestrator — Run skills by phase, name, or research pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from .paths import resolve_competition_paths

import pkgutil
import importlib
import types
import zindian.skills as skills_pkg

# Phase definitions (names correspond to module prefixes `skill_XX`)
# SoT v2.2.1 specifies 6 sub-phases: 1, 2A, 2B, 3A, 3B, 4
# skill_03 is split: policy_writer() runs in Phase 1, policy_gate() runs first in Phase 2A
PHASE_1_SKILLS = [
    "skill_01",
    "skill_02",
    "skill_03.policy_writer",
    "skill_04",
    "skill_05",
    "skill_15",
]
PHASE_2A_SKILLS = [
    "skill_03.policy_gate",
    "skill_06",
]  # Policy gate runs FIRST, then data cleaning
PHASE_2B_SKILLS = [
    "skill_07",
    "skill_08",
    "skill_07",
]  # Feature extraction, then anchor baseline, then variant training
PHASE_3A_SKILLS = ["skill_10", "skill_09", "skill_12"]  # Generalization audit
PHASE_3B_SKILLS = ["skill_11", "skill_21", "skill_13"]  # Promotion and fusion
PHASE_4_SKILLS = ["skill_14", "skill_16", "skill_17", "skill_22"]  # Governance


def _discover_skills() -> Dict[str, tuple[str, Optional[types.ModuleType]]]:
    """Dynamically discover and import modules under `zindian.skills`.

    Returns a mapping from skill key (e.g., 'skill_01') to a tuple of
    (description, module) where module may be None if import failed.
    """
    registry: Dict[str, tuple[str, Optional[types.ModuleType]]] = {}
    for finder, name, ispkg in pkgutil.iter_modules(skills_pkg.__path__):
        if not name.startswith("skill_"):
            continue
        full_name = f"zindian.skills.{name}"
        try:
            mod = importlib.import_module(full_name)
            desc = (
                (mod.__doc__ or "").strip().splitlines()[0]
                if getattr(mod, "__doc__", None)
                else name
            )
            registry[name] = (desc, mod)

            # Map prefix like 'skill_01' to registry
            import re

            m = re.match(r"^(skill_\d+)", name)
            if m:
                prefix = m.group(1)
                if prefix in registry:
                    # Resolve precedence for dual-file skills
                    existing_mod = registry[prefix][1]
                    _ = existing_mod.__name__.split(".")[-1] if existing_mod else ""
                    if prefix == "skill_13" and name == "skill_13_oracle_fusion":
                        registry[prefix] = (desc, mod)
                    elif prefix == "skill_00" and name == "skill_00_zindi_monitor":
                        registry[prefix] = (desc, mod)
                else:
                    registry[prefix] = (desc, mod)
        except Exception:
            registry[name] = (name, None)
            import re

            m = re.match(r"^(skill_\d+)", name)
            if m:
                prefix = m.group(1)
                if prefix not in registry:
                    registry[prefix] = (name, None)
    return registry


# Build registry at import time
SKILL_REGISTRY = _discover_skills()


def _validate_phase_map() -> None:
    """Check that any skills declared in challenge_config.phase_skill_map exist in SKILL_REGISTRY.

    Prints warnings for any missing skills so maintainers can fix config or add shims.
    """
    try:
        from .config import ChallengeConfig

        cfg = ChallengeConfig.load()
        phase_map = cfg.get("phase_skill_map", {}) or {}
    except Exception:
        phase_map = {}

    missing = []
    for phase, skills in phase_map.items():
        for s in skills:
            if s not in SKILL_REGISTRY:
                missing.append((phase, s))

    if missing:
        print(
            "[orchestrator] WARNING: phase_skill_map contains skills not discovered in SKILL_REGISTRY:"
        )
        for phase, s in missing:
            print(f"  - phase {phase}: {s}")
        print(
            "[orchestrator] Please ensure skill modules exist or update challenge_config.json."
        )


# Validate at import time so misconfigurations are visible early
_validate_phase_map()


def run_deep_research(
    domain: str = "geospatial",
    dry_run: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Skills 18, 19, and 20 asynchronously as a non-blocking background daemon."""
    import threading

    def _bg_run() -> None:
        try:
            paths = resolve_competition_paths(require_competition=True)
            reports_dir = paths.reports_dir
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Lookup deep research skills from registry
            lib_desc, lib_mod = SKILL_REGISTRY.get("skill_18", (None, None))
            miner_desc, miner_mod = SKILL_REGISTRY.get("skill_19", (None, None))
            sci_desc, sci_mod = SKILL_REGISTRY.get("skill_20", (None, None))

            if lib_mod is None or miner_mod is None or sci_mod is None:
                print("[deep_research] Error: sidecar skills are not loaded")
                return

            literature_cache_path = reports_dir / "literature_cache.json"
            domain_hypotheses_path = reports_dir / "domain_hypotheses.json"
            priorart_path = reports_dir / "ml_priorart.json"
            validated_hypotheses_path = reports_dir / "validated_hypotheses.json"
            failed_hypotheses_path = reports_dir / "failed_hypotheses.json"

            print("[deep_research] Starting background Librarian (Skill 18)...")
            lib_mod.run_librarian(
                config_path=str(paths.config_path),
                cache_path=str(literature_cache_path),
            )

            print("[deep_research] Starting background Code Miner (Skill 19)...")
            miner_mod.run_code_miner(
                domain=domain,
                dry_run=dry_run,
            )

            print("[deep_research] Starting background Scientist (Skill 20)...")
            sci_mod.run_scientist(
                hypotheses_path=str(domain_hypotheses_path),
                priorart_path=str(priorart_path),
                hypothesis_path=str(validated_hypotheses_path),
                failed_hypotheses_path=str(failed_hypotheses_path),
            )
            print("[deep_research] Background deep research flow complete.")
        except Exception as bg_exc:
            print(
                f"[deep_research] Background execution encountered exception: {bg_exc}"
            )

    # Start the daemon thread so it runs in background and doesn't block the main thread
    bg_thread = threading.Thread(
        target=_bg_run, daemon=True, name="ZindianDeepResearchDaemon"
    )
    bg_thread.start()

    paths = resolve_competition_paths(require_competition=True)
    reports_dir = paths.reports_dir
    return {
        "status": "LAUNCHED",
        "message": "Deep research sidecar launched in non-blocking background daemon thread.",
        "paths": {
            "literature_cache": str(reports_dir / "literature_cache.json"),
            "domain_hypotheses": str(reports_dir / "domain_hypotheses.json"),
            "priorart": str(reports_dir / "ml_priorart.json"),
            "validated_hypotheses": str(reports_dir / "validated_hypotheses.json"),
            "failed_hypotheses": str(reports_dir / "failed_hypotheses.json"),
        },
        **kwargs,
    }


def run_skill(
    skill_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run a single skill by name.

    Args:
        skill_name: e.g., "skill_01", "skill_02", "skill_03.policy_writer"
        **kwargs: Arguments to pass to the skill's run() function

    Returns:
        Result dict from skill
    """
    import time
    import tracemalloc

    # Start telemetry
    start_time = time.time()
    tracemalloc.start()

    # Handle split function notation (e.g., "skill_03.policy_writer")
    if "." in skill_name:
        base_skill, func_name = skill_name.split(".", 1)
        if base_skill not in SKILL_REGISTRY:
            return {
                "status": "ERROR",
                "message": f"Unknown skill: {base_skill}. Available: {list(SKILL_REGISTRY.keys())}",
            }

        description, skill_module = SKILL_REGISTRY[base_skill]

        if skill_module is None:
            return {
                "status": "ERROR",
                "message": f"Skill {base_skill} ({description}) not loaded",
            }

        # Call the specific function
        if not hasattr(skill_module, func_name):
            return {
                "status": "ERROR",
                "message": f"Skill {base_skill} has no function {func_name}",
            }

        try:
            func = getattr(skill_module, func_name)
            import inspect

            sig = inspect.signature(func)
            has_var_keyword = any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            )
            filtered_kwargs = (
                kwargs
                if has_var_keyword
                else {k: v for k, v in kwargs.items() if k in sig.parameters}
            )
            result = func(**filtered_kwargs)
        except Exception as e:
            import traceback
            from .config import ConfigNotPopulated

            # Graceful diagnostics for configuration errors
            if isinstance(e, ConfigNotPopulated):
                print(f"\n[orchestrator] ⚠️  CONFIGURATION ERROR: {str(e)}")
            elif isinstance(e, KeyError):
                print(f"\n[orchestrator] ⚠️  CONFIGURATION ERROR: Missing key '{e}'")

            result = {
                "status": "ERROR",
                "message": f"Skill {skill_name} failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }
    else:
        # Standard skill execution — skill_02 needs merge mode to preserve pre-set config
        if skill_name == "skill_02":
            kwargs.setdefault("merge", True)
        if skill_name not in SKILL_REGISTRY:
            return {
                "status": "ERROR",
                "message": f"Unknown skill: {skill_name}. Available: {list(SKILL_REGISTRY.keys())}",
            }

        description, skill_module = SKILL_REGISTRY[skill_name]

        if skill_module is None:
            return {
                "status": "ERROR",
                "message": f"Skill {skill_name} ({description}) not loaded",
            }

        try:
            import inspect

            sig = inspect.signature(skill_module.run)
            has_var_keyword = any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            )
            filtered_kwargs = (
                kwargs
                if has_var_keyword
                else {k: v for k, v in kwargs.items() if k in sig.parameters}
            )
            result = skill_module.run(**filtered_kwargs)
        except Exception as e:
            import traceback
            from .config import ConfigNotPopulated

            # Graceful diagnostics for configuration errors
            if isinstance(e, ConfigNotPopulated):
                print(f"\n[orchestrator] ⚠️  CONFIGURATION ERROR: {str(e)}")
            elif isinstance(e, KeyError):
                print(f"\n[orchestrator] ⚠️  CONFIGURATION ERROR: Missing key '{e}'")

            result = {
                "status": "ERROR",
                "message": f"Skill {skill_name} failed: {str(e)}",
                "traceback": traceback.format_exc(),
            }

    # Stop telemetry
    duration_sec = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    # R5: Carbon tracking
    carbon_data = {}
    try:
        from .carbon_tracker import estimate_carbon
        from .config import ChallengeConfig

        config_obj = ChallengeConfig.load()
        carbon_data = estimate_carbon(duration_sec, peak_memory_mb, config_obj._data)
    except Exception:
        carbon_data = {
            "carbon_kg_estimate": None,
            "tracker_method": "not_instrumented",
            "hardware_type": "unknown",
            "region": "unknown",
        }

    # Normalize result
    if result is None or not isinstance(result, dict):
        result = {"status": "COMPLETED"}
    elif "status" not in result:
        result["status"] = "COMPLETED"

    # Add telemetry with carbon data
    result_dict = cast(dict[str, Any], result)
    result_dict["telemetry"] = {
        "duration_sec": round(duration_sec, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        **carbon_data,
    }

    return result_dict


def prompt_human_gate(
    gate_name: str,
    store: Any,
    state: Dict[str, Any],
    config: Any,
    variant_name: Optional[str] = None,
    non_interactive: bool = False,
) -> bool:
    import json
    import sys
    from pathlib import Path
    from datetime import datetime, timezone

    # 1. Non-interactive check
    if non_interactive or not sys.stdin.isatty():
        if gate_name == "Gate 1":
            return bool(state.get("human_gate_1_approved"))
        elif gate_name == "Gate 2":
            return bool(state.get(f"human_gate_2_{variant_name}_approved"))
        elif gate_name == "Gate 3":
            return bool(state.get("human_gate_3_approved"))
        elif gate_name == "Gate 4":
            return bool(state.get("human_gate_4_approved"))
        return False

    # 2. Check if already approved
    if gate_name == "Gate 1" and state.get("human_gate_1_approved"):
        return True
    if gate_name == "Gate 2" and state.get(f"human_gate_2_{variant_name}_approved"):
        return True
    if gate_name == "Gate 3" and state.get("human_gate_3_approved"):
        return True
    if gate_name == "Gate 4" and state.get("human_gate_4_approved"):
        return True

    # 3. Interactive prompt
    print("\n============================================================")
    print(f"HUMAN GATE CONTROL: {gate_name.upper()}")
    if variant_name:
        print(f"Variant: {variant_name}")
    print("============================================================\n")

    if gate_name == "Gate 1":
        auto_strategy = config._data.get("cv_strategy", {}).get("type", "unknown")
        supports_d = auto_strategy in ("TimeSeriesSplit", "GroupKFold")
        print(f"Auto-selected CV strategy: {auto_strategy}")
        print(
            "Please review the anchor fold scores in the DuckDB ledger / status command."
        )
        print()
        while True:
            print("  [A] APPROVE  — accept auto-selected strategy")
            print("  [B] REJECT   — reject anchor, regenerate")
            print("  [C] CHALLENGE — override anchor inputs")
            if supports_d:
                print("  [D] CHALLENGE CV STRATEGY — comparison run")
            print()
            try:
                choice = input("Enter choice: ").strip().upper()
            except (KeyboardInterrupt, EOFError):
                print("\nAborted by user.")
                sys.exit(0)

            if choice == "A":
                store.update(human_gate_1_approved=True)
                print("✓ Gate 1 approved.")
                return True
            elif choice == "B":
                print("✗ Anchor rejected.")
                return False
            elif choice == "C":
                override_path = input("Enter path to JSON override file: ").strip()
                if not override_path:
                    print("❌ Override path cannot be empty.")
                    continue
                override_file = Path(override_path)
                if not override_file.exists():
                    print(f"❌ Override file not found: {override_file}")
                    continue
                try:
                    overrides = json.loads(override_file.read_text(encoding="utf-8"))
                except Exception as e:
                    print(f"❌ Failed to parse override JSON: {e}")
                    continue

                # Run challenge anchor
                original_oof = state.get("anchor_oof_score")
                model_family = (
                    overrides.get("model_family")
                    or overrides.get("framework")
                    or "lightgbm"
                )
                params = overrides.get("params") or overrides.get("hyperparams") or {}
                n_splits = overrides.get("n_splits") or config._data.get(
                    "cv_strategy", {}
                ).get("n_splits", 5)

                challenge_meta = {
                    "active": True,
                    "model_family": model_family,
                    "params": params,
                    "n_splits": n_splits,
                    "modification": f"Hyperparameters overridden from {override_file.name}",
                    "original_oof": original_oof,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                store.update(anchor_challenge=challenge_meta)

                # Execute challenge run
                print("\nRunning challenge anchor training...")
                result = run_skill("skill_08")

                fresh_state = store.read()
                challenged_oof = fresh_state.get("anchor_oof_score")

                challenge_meta["challenged_oof"] = challenged_oof
                challenge_meta["approved_by"] = "human_gate_1"

                print("\n--- CHALLENGE RUN COMPLETE ---")
                print(f"Original Anchor OOF Score : {original_oof}")
                print(f"Challenged Anchor OOF Score: {challenged_oof}")
                print()

                while True:
                    sel = (
                        input(
                            "Choose which anchor to keep ([O]riginal / [C]hallenged): "
                        )
                        .strip()
                        .upper()
                    )
                    if sel == "C":
                        challenge_meta["active"] = True
                        challenge_meta["rationale"] = "User chose challenged anchor"
                        store.update(
                            anchor_oof_score=challenged_oof,
                            anchor_oof_score_challenged=challenged_oof,
                            anchor_challenge=challenge_meta,
                            human_gate_1_approved=True,
                        )
                        print("✓ Challenged anchor accepted. Gate 1 approved.")
                        return True
                    elif sel == "O":
                        challenge_meta["active"] = False
                        challenge_meta["rationale"] = "User retained original anchor"
                        store.update(
                            anchor_oof_score=original_oof,
                            anchor_challenge=challenge_meta,
                            human_gate_1_approved=True,
                        )
                        print("✓ Original anchor retained. Gate 1 approved.")
                        return True
                    else:
                        print("❌ Invalid selection. Enter 'O' or 'C'.")
            elif choice == "D" and supports_d:
                task_type = config._data.get("task_type", "regression")
                override_strategy = (
                    "StratifiedKFold" if task_type == "classification" else "KFold"
                )
                original_oof = state.get("anchor_oof_score")

                override_meta = {
                    "active": True,
                    "original_strategy": auto_strategy,
                    "override_strategy": override_strategy,
                    "original_oof": original_oof,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                store.update(cv_strategy_override=override_meta)

                print(
                    f"\nRunning comparison CV strategy training using {override_strategy}..."
                )
                result = run_skill("skill_08")

                fresh_state = store.read()
                override_oof = fresh_state.get("anchor_oof_score")

                override_meta["override_oof"] = override_oof
                override_meta["approved_by"] = "human_gate_1"

                print("\n--- CV Strategy COMPARISON RUN COMPLETE ---")
                print(f"Original Strategy ({auto_strategy}) OOF: {original_oof}")
                print(f"Override Strategy ({override_strategy}) OOF: {override_oof}")
                print()

                while True:
                    sel = (
                        input(
                            "Choose which CV strategy to use ([O]riginal / [C]omparison): "
                        )
                        .strip()
                        .upper()
                    )
                    if sel == "C":
                        override_meta["active"] = True
                        override_meta["rationale"] = "User chose comparison CV strategy"
                        store.update(
                            anchor_oof_score=override_oof,
                            cv_strategy_override=override_meta,
                            human_gate_1_approved=True,
                        )
                        print(
                            f"✓ CV strategy override ({override_strategy}) accepted. Gate 1 approved."
                        )
                        return True
                    elif sel == "O":
                        override_meta["active"] = False
                        override_meta["rationale"] = (
                            "User retained original CV strategy"
                        )
                        store.update(
                            anchor_oof_score=original_oof,
                            cv_strategy_override=override_meta,
                            human_gate_1_approved=True,
                        )
                        print("✓ Original CV strategy retained. Gate 1 approved.")
                        return True
                    else:
                        print("❌ Invalid selection. Enter 'O' or 'C'.")
            else:
                print("❌ Invalid selection.")

    elif gate_name == "Gate 2":
        while True:
            try:
                choice = (
                    input(f"Approve variant branch '{variant_name}'? [YES/NO]: ")
                    .strip()
                    .upper()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nAborted by user.")
                sys.exit(0)
            if choice == "YES":
                store.update(**{f"human_gate_2_{variant_name}_approved": True})
                print(f"✓ Variant '{variant_name}' approved.")
                return True
            elif choice == "NO":
                print(f"✗ Variant '{variant_name}' rejected.")
                return False
            else:
                print("❌ Enter YES or NO.")

    elif gate_name == "Gate 3":
        while True:
            try:
                choice = (
                    input("Approve entering oracle fusion? [YES/NO]: ").strip().upper()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nAborted by user.")
                sys.exit(0)
            if choice == "YES":
                store.update(human_gate_3_approved=True)
                print("✓ Gate 3 approved.")
                return True
            elif choice == "NO":
                print("✗ Fusion rejected.")
                return False
            else:
                print("❌ Enter YES or NO.")

    elif gate_name == "Gate 4":
        while True:
            try:
                choice = (
                    input("Approve generating inference predictions? [YES/NO]: ")
                    .strip()
                    .upper()
                )
            except (KeyboardInterrupt, EOFError):
                print("\nAborted by user.")
                sys.exit(0)
            if choice == "YES":
                store.update(human_gate_4_approved=True)
                print("✓ Gate 4 approved.")
                return True
            elif choice == "NO":
                print("✗ Inference prediction generation rejected.")
                return False
            else:
                print("❌ Enter YES or NO.")

    return False


def run_phase(
    phase: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run all skills for a given phase.

    Args:
        phase: "1", "2A", "2B", "3A", "3B", or "4"
        **kwargs: Arguments to pass to each skill's run() function

    Returns:
        Dict with results for each skill
    """
    # Load config and state for skills that need them
    try:
        from .config import ChallengeConfig
        from .state import SkillStateStore
        from .paths import resolve_competition_paths

        paths = resolve_competition_paths(require_competition=False)
        config = ChallengeConfig.load()
        store = SkillStateStore(paths.state_path)
        state = store.read()
    except Exception:
        config = None
        state = {}
        store = None
        paths = None

    # Enforce phase dependencies (Principle 4)
    try:
        if not paths:
            from .paths import resolve_competition_paths

            paths = resolve_competition_paths(require_competition=False)
        if not store and paths.state_path.exists():
            from .state import SkillStateStore

            store = SkillStateStore(paths.state_path)
            state = store.read()

            # Phase dependency checks - block execution if prerequisites not met
            if phase == "2A":
                if (
                    not state.get("phase_1_complete")
                    and state.get("dag_phase") != "phase_1_complete"
                ):
                    return {
                        "status": "ERROR",
                        "message": "Phase 2A blocked: Phase 1 must complete first",
                        "required": "phase_1_complete",
                    }
            elif phase == "2B":
                if not state.get("phase_2a_complete"):
                    return {
                        "status": "ERROR",
                        "message": "Phase 2B blocked: Phase 2A must complete first",
                        "required": "phase_2a_complete",
                    }
            elif phase == "3A":
                if not state.get("phase_2b_complete"):
                    return {
                        "status": "ERROR",
                        "message": "Phase 3A blocked: Phase 2B must complete first",
                        "required": "phase_2b_complete",
                    }
            elif phase == "3B":
                if not state.get("phase_3a_complete"):
                    return {
                        "status": "ERROR",
                        "message": "Phase 3B blocked: Phase 3A must complete first",
                        "required": "phase_3a_complete",
                    }
            elif phase == "4":
                if not state.get("phase_3b_complete"):
                    return {
                        "status": "ERROR",
                        "message": "Phase 4 blocked: Phase 3B must complete first",
                        "required": "phase_3b_complete",
                    }
    except Exception:
        pass  # Allow execution if state check fails (INIT mode)

    # Prefer configured phase map if present; otherwise fall back to hardcoded lists
    try:
        from .config import ChallengeConfig

        cfg = ChallengeConfig.load()
        phase_map = cfg.get("phase_skill_map", None)
    except Exception:
        phase_map = None

    if phase_map and phase in phase_map:
        skills = phase_map[phase]
    else:
        if phase == "1":
            skills = PHASE_1_SKILLS
        elif phase == "2A":
            skills = PHASE_2A_SKILLS
        elif phase == "2B":
            skills = PHASE_2B_SKILLS
        elif phase == "3A":
            skills = PHASE_3A_SKILLS
        elif phase == "3B":
            skills = PHASE_3B_SKILLS
        elif phase == "4":
            skills = PHASE_4_SKILLS
        else:
            return {
                "status": "ERROR",
                "message": f"Invalid phase: {phase}. Must be 1, 2A, 2B, 3A, 3B, or 4.",
            }

    results = {}
    for skill_name in skills:
        variant_arg = kwargs.get("variant_name")
        if phase == "2B" and variant_arg and skill_name == "skill_08":
            print(
                f"\nSkipping {skill_name} (anchor baseline training) for variant run: {variant_arg}\n"
            )
            continue

        if skill_name in SKILL_REGISTRY or "." in skill_name:
            # --- Human Gates Intercepts ---
            non_interactive = kwargs.get("non_interactive", False)

            if store:
                try:
                    state = store.read()
                except Exception:
                    pass

            # Gate 1 Check: at start of variant run, check Gate 1
            if (
                phase == "2B"
                and variant_arg
                and skill_name == "skill_07"
                and not state.get("human_gate_1_approved")
            ):
                approved = prompt_human_gate(
                    "Gate 1",
                    store,
                    state,
                    config,
                    variant_name=variant_arg,
                    non_interactive=non_interactive,
                )
                if not approved:
                    return {
                        "status": "ERROR",
                        "message": "Phase 2B variant execution blocked: Gate 1 not approved",
                    }

            # Gate 3 Check: before skill_13 runs
            if phase == "3B" and skill_name == "skill_13":
                approved = prompt_human_gate(
                    "Gate 3", store, state, config, non_interactive=non_interactive
                )
                if not approved:
                    return {
                        "status": "ERROR",
                        "message": "Phase 3B fusion execution blocked: Gate 3 not approved",
                    }

            # Gate 4 Check: before skill_14 runs
            if phase == "4" and skill_name == "skill_14":
                approved = prompt_human_gate(
                    "Gate 4", store, state, config, non_interactive=non_interactive
                )
                if not approved:
                    return {
                        "status": "ERROR",
                        "message": "Phase 4 inference execution blocked: Gate 4 not approved",
                    }

            print(f"\nRunning {skill_name}...")
            # Pass config and state to skills that need them
            skill_kwargs = kwargs.copy()
            if skill_name == "skill_03.policy_gate":
                # Load policy and planned_features for policy_gate
                try:
                    import json

                    if paths is not None:
                        policy_path = paths.reports_dir / "feature_policy.json"
                        if policy_path.exists():
                            policy = json.loads(policy_path.read_text())
                            skill_kwargs["policy"] = policy
                    skill_kwargs["planned_features"] = state.get("planned_features", [])
                    pass
                except Exception:
                    pass
            elif skill_name == "skill_03.policy_writer":
                skill_kwargs["monitor_data"] = (
                    state.get("monitor_data", state.get("community_signals", {})) or {}
                )
                skill_kwargs["config"] = config._data if config else {}
                skill_kwargs["flagged_titles"] = state.get("flagged_titles", []) or []
            elif skill_name == "skill_06":
                # Materialize feature matrices for skill_06
                import pandas as pd

                if paths is not None:
                    train_path = paths.data_raw_dir / "Train.csv"
                    test_path = paths.data_raw_dir / "Test.csv"
                else:
                    from zindian.paths import resolve_competition_paths as _rcp

                    _p = _rcp()
                    train_path = _p.data_raw_dir / "Train.csv"
                    test_path = _p.data_raw_dir / "Test.csv"

                df_train = pd.read_csv(train_path)
                df_test = pd.read_csv(test_path)

                # Extract target columns dynamically from config
                target_cols = []
                id_col = config._data.get("id_col", "ID") if config else "ID"

                if config:
                    target_config = config._data.get("target_config")
                    if target_config and isinstance(target_config, dict):
                        targets = target_config.get("targets", [])
                        target_cols = [t["name"] for t in targets]
                    else:
                        # Fallback to single target
                        for key in ("target_col", "target", "target_column", "label"):
                            value = config._data.get(key)
                            if value:
                                target_cols.append(str(value))
                                break

                # Separate targets and create feature matrices
                X_train = df_train.drop(columns=[id_col] + target_cols, errors="ignore")
                X_test = df_test.drop(columns=[id_col], errors="ignore")

                # Pass to skill_06
                skill_kwargs["config"] = config._data if config else {}
                skill_kwargs["state"] = {**state, "X_train": X_train, "X_test": X_test}

            result = run_skill(skill_name, **skill_kwargs)
            results[skill_name] = result

            # --- Post-Skill human gate checks ---
            if store:
                try:
                    state = store.read()
                except Exception:
                    pass

            # Gate 1 Check: after skill_08 completes in anchor run (no variant)
            if phase == "2B" and not variant_arg and skill_name == "skill_08":
                approved = prompt_human_gate(
                    "Gate 1", store, state, config, non_interactive=non_interactive
                )
                if not approved:
                    return {
                        "status": "ERROR",
                        "message": "Phase 2B anchor execution blocked: Gate 1 not approved",
                    }

            # Gate 2 Check: after skill_08 completes in variant run
            if phase == "2B" and variant_arg and skill_name == "skill_08":
                approved = prompt_human_gate(
                    "Gate 2",
                    store,
                    state,
                    config,
                    variant_name=variant_arg,
                    non_interactive=non_interactive,
                )
                if not approved:
                    return {
                        "status": "ERROR",
                        "message": f"Phase 2B variant '{variant_arg}' execution blocked: Gate 2 not approved",
                    }

            # Print telemetry
            telemetry = result.get("telemetry", {})
            if telemetry:
                print(
                    f"  Time: {telemetry.get('duration_sec', 0)}s | Peak RAM: {telemetry.get('peak_memory_mb', 0)}MB"
                )

            # Store telemetry in state
            try:
                if store:
                    telemetry_key = f"telemetry.{skill_name}"
                    store.update(**{telemetry_key: telemetry})
            except Exception:
                pass
        else:
            results[skill_name] = {
                "status": "SKIPPED",
                "message": f"Skill {skill_name} not yet implemented",
            }

    # Mark phase complete in state
    try:
        from .state import SkillStateStore
        from .paths import resolve_competition_paths

        paths = resolve_competition_paths(require_competition=False)
        if paths.state_path.exists():
            store = SkillStateStore(paths.state_path)
            phase_key = (
                f"phase_{phase.lower().replace('a', 'a').replace('b', 'b')}_complete"
            )
            store.update(**{phase_key: True})
    except Exception:
        pass

    # Generate phase summary report
    try:
        from .skills.skill_15_reporter import run_phase_summary, _write_json_summary

        _phase = phase.lower().strip()
        if _phase in ("2b", "3b"):
            run_phase_summary(_phase)
        _write_json_summary(
            _phase,
            paths,
            state,
            [
                "anchor_oof_score",
                "best_variant_features",
                "submissions_used_total",
                "cv_strategy_type",
            ],
        )
    except Exception:
        pass

    return results
