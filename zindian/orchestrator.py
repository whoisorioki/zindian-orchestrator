"""Skill Orchestrator — Run skills by phase or name"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Import all skills
try:
    from .skills import skill_01_integrity, skill_15_reporter
    from .skills import skill_02_intake_new as skill_02_intake
except ImportError:
    skill_01_integrity = None
    skill_02_intake = None
    skill_15_reporter = None


PHASE_1_SKILLS = ["skill_01", "skill_02", "skill_15"]
PHASE_2_SKILLS = ["skill_03", "skill_08"]
PHASE_3_SKILLS = ["skill_04", "skill_05", "skill_09", "skill_10"]
PHASE_4_SKILLS = ["skill_11", "skill_16"]
PHASE_5_SKILLS = ["skill_13", "skill_14", "skill_17"]

SKILL_REGISTRY = {
    "skill_01": ("Integrity Audit", skill_01_integrity),
    "skill_02": ("Challenge Intake", skill_02_intake),
    "skill_15": ("Reporter", skill_15_reporter),
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
) -> Dict[str, list]:
    """
    Run all skills for a given phase.
    
    Args:
        phase: 1, 2, 3, 4, or 5
        **kwargs: Arguments to pass to each skill's run() function
    
    Returns:
        Dict with results for each skill
    """
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
