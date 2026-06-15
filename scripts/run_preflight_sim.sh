#!/usr/bin/env bash
# run_preflight_sim.sh
# Preflight simulation — dual-target validation.
# tmpcomp should PASS. ey-frogs should FAIL with known structural anomalies.
#
# Usage:
#   bash scripts/run_preflight_sim.sh
#
# Expected ey-frogs failures (each one is a spec violation, not a bug here):
#   - anchor_oof_score null if Phase 2 not yet run (WARN only, non-blocking)
#   - Any config fields absent from challenge_config.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PREFLIGHT="$REPO_ROOT/scripts/preflight_enforce.py"

echo "============================================"
echo " Preflight Sim A — tmpcomp (expect PASS)"
echo "============================================"
if [ -d "$REPO_ROOT/competitions/tmpcomp" ]; then
    python "$PREFLIGHT" --competition "$REPO_ROOT/competitions/tmpcomp"
else
    echo "SKIP: competitions/tmpcomp not found — create a minimal fixture to run this target"
fi
echo ""

echo "============================================"
echo " Preflight Sim B — ey-frogs (expect FAIL)"
echo " Known structural anomalies expected in output:"
echo "   - missing required config keys (if any)"
echo "   - anchor_oof_score null (WARN) if Phase 2 not run"
echo "============================================"
if [ -d "$REPO_ROOT/competitions/ey-frogs" ]; then
    python "$PREFLIGHT" --competition "$REPO_ROOT/competitions/ey-frogs" || true
    # '|| true' prevents the script from exiting on expected preflight failure
else
    echo "SKIP: competitions/ey-frogs not found"
fi
echo ""
echo "Preflight sim complete."
echo "Review output above to confirm known failures match expectations."
