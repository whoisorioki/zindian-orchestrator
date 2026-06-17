"""
Simple SKILL_STATE wrapper - use everywhere instead of json.load/dump
"""
import json
from pathlib import Path

# Add to your imports:
# from tabula.skill_state import read_state, write_state

def read_state(path):
    """Read SKILL_STATE.json (auto-loads externalized scores)."""
    path = Path(path)
    with open(path) as f:
        state = json.load(f)
    
    # Load externalized scores
    for key, value in state.items():
        if isinstance(value, dict) and "scores_file" in value:
            score_path = path.parent / value["scores_file"]
            if score_path.exists():
                with open(score_path) as sf:
                    value["scores"] = json.load(sf)
    
    return state


def write_state(path, state):
    """Write SKILL_STATE.json (auto-externalizes large score arrays)."""
    path = Path(path)
    scores_dir = path.parent / "scores"
    scores_dir.mkdir(exist_ok=True)
    
    state_copy = {}
    for key, value in state.items():
        if isinstance(value, dict) and "scores" in value:
            scores = value["scores"]
            if isinstance(scores, list) and len(scores) > 100:
                # Externalize large arrays
                score_file = scores_dir / f"{key}.json"
                with open(score_file, 'w') as sf:
                    json.dump(scores, sf)
                
                # Keep other fields, replace scores with reference
                new_value = {k: v for k, v in value.items() if k != "scores"}
                new_value["scores_file"] = f"scores/{key}.json"
                new_value["count"] = len(scores)
                state_copy[key] = new_value
            else:
                state_copy[key] = value
        else:
            state_copy[key] = value
    
    with open(path, 'w') as f:
        json.dump(state_copy, f, indent=2)
