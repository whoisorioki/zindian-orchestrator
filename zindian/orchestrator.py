"""Skill Orchestrator — Run skills by phase, name, or research pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .paths import resolve_competition_paths

import pkgutil
import importlib
import types
import zindian.skills as skills_pkg


# Phase definitions (names correspond to module prefixes `skill_XX`)
PHASE_1_SKILLS = ["skill_01", "skill_02", "skill_15"]
PHASE_2_SKILLS = ["skill_03", "skill_08"]
PHASE_3_SKILLS = ["skill_04", "skill_05", "skill_09", "skill_10"]
PHASE_4_SKILLS = ["skill_11", "skill_16"]
PHASE_5_SKILLS = ["skill_13", "skill_14", "skill_17"]


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
            desc = (mod.__doc__ or "").strip().splitlines()[0] if getattr(mod, "__doc__", None) else name
            registry[name] = (desc, mod)
        except Exception:
            registry[name] = (name, None)
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
        print("[orchestrator] WARNING: phase_skill_map contains skills not discovered in SKILL_REGISTRY:")
        for phase, s in missing:
            print(f"  - phase {phase}: {s}")
        print("[orchestrator] Please ensure skill modules exist or update challenge_config.json.")


# Validate at import time so misconfigurations are visible early
_validate_phase_map()


def run_deep_research(
    domain: str = "geospatial",
    dry_run: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run Skills 18, 19, and 20 in the intended research order."""
    paths = resolve_competition_paths(require_competition=True)
    reports_dir = paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Lookup deep research skills from registry
    lib_desc, lib_mod = SKILL_REGISTRY.get("skill_18", (None, None))
    miner_desc, miner_mod = SKILL_REGISTRY.get("skill_19", (None, None))
    sci_desc, sci_mod = SKILL_REGISTRY.get("skill_20", (None, None))

    if lib_mod is None or miner_mod is None or sci_mod is None:
        return {
            "status": "ERROR",
            "message": "Deep research skills are not loaded",
        }

    literature_cache_path = reports_dir / "literature_cache.json"
    domain_hypotheses_path = reports_dir / "domain_hypotheses.json"
    priorart_path = reports_dir / "ml_priorart.json"
    validated_hypotheses_path = reports_dir / "validated_hypotheses.json"
    failed_hypotheses_path = reports_dir / "failed_hypotheses.json"

    librarian_result = lib_mod.run_librarian(
        config_path=str(paths.config_path),
        cache_path=str(literature_cache_path),
    )

    code_miner_result = miner_mod.run_code_miner(
        domain=domain,
        dry_run=dry_run,
    )

    scientist_result = sci_mod.run_scientist(
        hypotheses_path=str(domain_hypotheses_path),
        priorart_path=str(priorart_path),
        hypothesis_path=str(validated_hypotheses_path),
        failed_hypotheses_path=str(failed_hypotheses_path),
    )

    return {
        "status": "OK",
        "librarian": librarian_result,
        "code_miner": code_miner_result,
        "scientist": scientist_result,
        "paths": {
            "literature_cache": str(literature_cache_path),
            "domain_hypotheses": str(domain_hypotheses_path),
            "priorart": str(priorart_path),
            "validated_hypotheses": str(validated_hypotheses_path),
            "failed_hypotheses": str(failed_hypotheses_path),
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
        skill_name: e.g., "skill_01", "skill_02", "skill_15"
        **kwargs: Arguments to pass to the skill's run() function
    
    Returns:
        Result dict from skill
    """
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
        return skill_module.run(**kwargs)
    except Exception as e:
        import traceback
        return {
            "status": "ERROR",
            "message": f"Skill {skill_name} failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def run_phase(
    phase: int,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run all skills for a given phase.
    
    Args:
        phase: 1, 2, 3, 4, or 5
        **kwargs: Arguments to pass to each skill's run() function
    
    Returns:
        Dict with results for each skill
    """
    # Prefer configured phase map if present; otherwise fall back to hardcoded lists
    try:
        from .config import ChallengeConfig
        cfg = ChallengeConfig.load()
        phase_map = cfg.get("phase_skill_map", None)
    except Exception:
        phase_map = None

    if phase_map and str(phase) in phase_map:
        skills = phase_map[str(phase)]
    else:
        if phase == 1:
            skills = PHASE_1_SKILLS
        elif phase == 2:
            skills = PHASE_2_SKILLS
        elif phase == 3:
            skills = PHASE_3_SKILLS
        elif phase == 4:
            skills = PHASE_4_SKILLS
        elif phase == 5:
            skills = PHASE_5_SKILLS
        else:
            return {
                "status": "ERROR",
                "message": f"Invalid phase: {phase}. Must be 1-5.",
            }
    
    results = {}
    for skill_name in skills:
        if skill_name in SKILL_REGISTRY:
            print(f"\nRunning {skill_name}...")
            results[skill_name] = run_skill(skill_name, **kwargs)
        else:
            results[skill_name] = {
                "status": "SKIPPED",
                "message": f"Skill {skill_name} not yet implemented",
            }
    
    return results
