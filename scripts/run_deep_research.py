"""
Run the research sidecar pipeline synchronously (Skills 18 → 19 → 20).
Calls each skill's specific entry point directly in the correct sequence,
blocking until all three complete. Writes output to reports/.
"""

import sys

sys.path.insert(0, ".")

from zindian.paths import resolve_competition_paths
from zindian.orchestrator import SKILL_REGISTRY

paths = resolve_competition_paths(require_competition=True)
reports_dir = paths.reports_dir
reports_dir.mkdir(parents=True, exist_ok=True)

literature_cache_path = str(reports_dir / "literature_cache.json")
domain_hypotheses_path = str(reports_dir / "domain_hypotheses.json")
priorart_path = str(reports_dir / "ml_priorart.json")
validated_hypotheses_path = str(reports_dir / "validated_hypotheses.json")
failed_hypotheses_path = str(reports_dir / "failed_hypotheses.json")

_, lib_mod = SKILL_REGISTRY["skill_18"]
_, miner_mod = SKILL_REGISTRY["skill_19"]
_, sci_mod = SKILL_REGISTRY["skill_20"]

print("=" * 60)
print("SKILL 18 — The Librarian")
print("=" * 60)
lib_mod.run_librarian(
    config_path=str(paths.config_path),
    cache_path=literature_cache_path,
)
print("[OK] Skill 18 complete\n")

print("=" * 60)
print("SKILL 19 — The Code Miner")
print("=" * 60)
miner_mod.run_code_miner(
    domain="geospatial",
    dry_run=False,
)
print("[OK] Skill 19 complete\n")

print("=" * 60)
print("SKILL 20 — The Scientist")
print("=" * 60)
validated = sci_mod.run_scientist(
    hypotheses_path=domain_hypotheses_path,
    priorart_path=priorart_path,
    hypothesis_path=validated_hypotheses_path,
    failed_hypotheses_path=failed_hypotheses_path,
)
print(f"[OK] Skill 20 complete — {len(validated)} validated hypotheses\n")

print("=" * 60)
print("RESEARCH SIDECAR PIPELINE COMPLETE")
print(f"  validated_hypotheses -> {validated_hypotheses_path}")
print("=" * 60)
