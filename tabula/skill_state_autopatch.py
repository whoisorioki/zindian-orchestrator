"""
Auto-patch for SKILL_STATE.json operations.
Import this at the top of any script to automatically handle score externalization.

Usage:
    import tabula.skill_state_autopatch  # noqa
    # Now all SKILL_STATE.json operations are automatically optimized
"""
import json
import builtins
from pathlib import Path
from typing import Any

_original_open = builtins.open
_original_json_load = json.load
_original_json_dump = json.dump
_original_json_dumps = json.dumps


def _patched_open(file, mode='r', *args, **kwargs):
    """Intercept SKILL_STATE.json opens."""
    f = _original_open(file, mode, *args, **kwargs)
    
    # Mark if this is a SKILL_STATE file
    if isinstance(file, (str, Path)) and "SKILL_STATE.json" in str(file):
        f._is_skill_state = True
        f._skill_state_path = Path(file)
    else:
        f._is_skill_state = False
    
    return f


def _patched_json_load(fp, *args, **kwargs):
    """Auto-load externalized scores for SKILL_STATE."""
    data = _original_json_load(fp, *args, **kwargs)
    
    if getattr(fp, '_is_skill_state', False):
        path = getattr(fp, '_skill_state_path', None)
        if path:
            # Load externalized scores
            for key, value in data.items():
                if isinstance(value, dict) and "scores_file" in value:
                    score_path = path.parent / value["scores_file"]
                    if score_path.exists():
                        with _original_open(score_path, 'r') as sf:
                            value["scores"] = _original_json_load(sf)
    
    return data


def _patched_json_dump(obj, fp, *args, **kwargs):
    """Auto-externalize large scores for SKILL_STATE."""
    if getattr(fp, '_is_skill_state', False):
        path = getattr(fp, '_skill_state_path', None)
        if path and isinstance(obj, dict):
            scores_dir = path.parent / "scores"
            scores_dir.mkdir(exist_ok=True)
            
            obj_copy = {}
            for key, value in obj.items():
                if isinstance(value, dict) and "scores" in value:
                    scores = value["scores"]
                    if isinstance(scores, list) and len(scores) > 100:
                        # Externalize
                        score_file = scores_dir / f"{key}.json"
                        with _original_open(score_file, 'w') as sf:
                            _original_json_dump(scores, sf)
                        
                        # Create new dict without scores, add reference
                        new_value = {k: v for k, v in value.items() if k != "scores"}
                        new_value["scores_file"] = f"scores/{key}.json"
                        new_value["count"] = len(scores)
                        obj_copy[key] = new_value
                    else:
                        obj_copy[key] = value
                else:
                    obj_copy[key] = value
            
            return _original_json_dump(obj_copy, fp, *args, **kwargs)
    
    return _original_json_dump(obj, fp, *args, **kwargs)


# Apply patches
builtins.open = _patched_open
json.load = _patched_json_load
json.dump = _patched_json_dump

print("[OK] SKILL_STATE auto-optimization enabled")
