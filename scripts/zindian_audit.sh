#!/bin/bash
# zindian_audit.sh — Framework completeness audit
# Run from: ~/projects/zindian_orchestrator
# Purpose: Report what exists, what is missing, and what is empty

ROOT="$(pwd)"
echo "========================================"
echo " ZINDIAN FRAMEWORK AUDIT"
echo " $(date)"
echo " Root: $ROOT"
echo "========================================"

# ── 1. CORE ROOT FILES ─────────────────────
echo ""
echo "── 1. CORE ROOT FILES ──"
for f in AGENTS.md CLAUDE.md README.md setup.py .gitignore .env; do
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    echo "  ✅ $f ($lines lines)"
  else
    echo "  ❌ MISSING: $f"
  fi
done

# ── 2. AGENT INSTRUCTION FILES ─────────────
echo ""
echo "── 2. AGENT INSTRUCTION FILES ──"
declare -A agent_files=(
  [".cursor/rules/zindian.md"]="Cursor"
  [".github/instructions/zindian.md"]="Copilot"
  [".kiro/specs/zindian.md"]="Kiro"
  [".opencode/agents/zindian.md"]="OpenCode"
  [".windsurf/rules/zindian.md"]="Windsurf"
)
for path in "${!agent_files[@]}"; do
  tool="${agent_files[$path]}"
  if [ -f "$path" ]; then
    lines=$(wc -l < "$path")
    echo "  ✅ $tool — $path ($lines lines)"
  else
    echo "  ❌ MISSING: $tool — $path"
  fi
done

# ── 3. OPENCODE SKILLS ─────────────────────
echo ""
echo "── 3. OPENCODE SKILLS (.opencode/skills/) ──"
expected_skills=(
  "00_registry.md"
  "01_integrity_governance.md"
  "02_challenge_intake.md"
  "03_deep_research_legality.md"
  "04_violation_eda.md"
  "05_cv_architect.md"
  "06_feature_engineering.md"
  "07_shap_audit.md"
  "08_anchor_baseline.md"
  "09_granular_calibration.md"
  "10_leakage_check.md"
  "11_branch_generator.md"
  "12_metric_tradeoff.md"
  "13_fusion_candidates.md"
  "14_post_processing.md"
  "15_reporter.md"
  "16_critique.md"
  "17_governance.md"
)
for skill in "${expected_skills[@]}"; do
  path=".opencode/skills/$skill"
  if [ -f "$path" ]; then
    lines=$(wc -l < "$path")
    echo "  ✅ $skill ($lines lines)"
  else
    echo "  ❌ MISSING: $skill"
  fi
done

# ── 4. PYTHON PACKAGE ──────────────────────
echo ""
echo "── 4. PYTHON PACKAGE (zindian/) ──"
expected_py=(
  "zindian/__init__.py"
  "zindian/state.py"
  "zindian/config.py"
  "zindian/ledger.py"
  "zindian/zindi_client.py"
  "zindian/skills/skill_01_integrity.py"
  "zindian/skills/skill_02_intake.py"
  "zindian/skills/skill_04_eda.py"
  "zindian/skills/skill_05_cv.py"
  "zindian/skills/skill_06_cleaning.py"
  "zindian/skills/skill_07_features.py"
  "zindian/skills/skill_08_anchor.py"
  "zindian/skills/skill_09_calibration.py"
  "zindian/skills/skill_10_shap.py"
  "zindian/skills/skill_11_gate.py"
  "zindian/skills/skill_12_metric.py"
  "zindian/skills/skill_13_fusion.py"
  "zindian/skills/skill_14_inference.py"
  "zindian/skills/skill_15_reporter.py"
  "zindian/skills/skill_16_critique.py"
  "zindian/skills/skill_17_governance.py"
)
for f in "${expected_py[@]}"; do
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    if [ "$lines" -lt 5 ]; then
      echo "  ⚠️  STUB: $f ($lines lines)"
    else
      echo "  ✅ $f ($lines lines)"
    fi
  else
    echo "  ❌ MISSING: $f"
  fi
done

# ── 5. TABULA CLI ──────────────────────────
echo ""
echo "── 5. TABULA CLI (tabula/) ──"
for f in "tabula/__init__.py" "tabula/init.py"; do
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    if [ "$lines" -lt 5 ]; then
      echo "  ⚠️  STUB: $f ($lines lines)"
    else
      echo "  ✅ $f ($lines lines)"
    fi
  else
    echo "  ❌ MISSING: $f"
  fi
done

# ── 6. SPECS ───────────────────────────────
echo ""
echo "── 6. SPECS (specs/) ──"
for f in "specs/requirements.md" "specs/design.md" "specs/tasks.md"; do
  if [ -f "$f" ]; then
    lines=$(wc -l < "$f")
    echo "  ✅ $f ($lines lines)"
  else
    echo "  ❌ MISSING: $f"
  fi
done

# ── 7. TEMPLATES ───────────────────────────
echo ""
echo "── 7. TEMPLATES (templates/) ──"
ls -la templates/ 2>/dev/null || echo "  ❌ templates/ missing"

# ── 8. SCRIPTS ─────────────────────────────
echo ""
echo "── 8. SCRIPTS (scripts/) ──"
ls -la scripts/ 2>/dev/null || echo "  ❌ scripts/ missing"

# ── 9. VENV PACKAGES ───────────────────────
echo ""
echo "── 9. VENV — REQUIRED PACKAGES ──"
source .venv/bin/activate 2>/dev/null
required_pkgs=(
  "lightgbm" "pandas" "numpy" "scikit-learn"
  "shap" "duckdb" "PyGithub" "python-dotenv" "requests"
)
for pkg in "${required_pkgs[@]}"; do
  version=$(pip show "$pkg" 2>/dev/null | grep Version | awk '{print $2}')
  if [ -n "$version" ]; then
    echo "  ✅ $pkg==$version"
  else
    echo "  ❌ NOT INSTALLED: $pkg"
  fi
done

# Check zindi package specifically
zindi_ver=$(pip show zindi 2>/dev/null | grep Version | awk '{print $2}')
if [ -n "$zindi_ver" ]; then
  echo "  ✅ zindi==$zindi_ver"
else
  echo "  ❌ NOT INSTALLED: zindi (KameniAlexNea fork)"
fi

# ── 10. COMPETITION STATE ──────────────────
echo ""
echo "── 10. COMPETITION STATE (ey-frogs) ──"
state_file="competitions/ey-frogs/SKILL_STATE.json"
config_file="competitions/ey-frogs/challenge_config.json"

if [ -f "$state_file" ]; then
  echo "  ✅ SKILL_STATE.json found"
  python3 -c "
import json
with open('$state_file') as f:
    s = json.load(f)
print(f'     dag_phase       : {s.get(\"dag_phase\")}')
print(f'     competition     : {s.get(\"competition\")}')
print(f'     md5_target_hash : {\"locked\" if s.get(\"md5_target_hash\") else \"null\"}')
print(f'     anchor_oof_rmse : {s.get(\"anchor_oof_rmse\")}')
print(f'     current_branch  : {s.get(\"current_git_branch\")}')
" 2>/dev/null
else
  echo "  ❌ MISSING: $state_file"
fi

if [ -f "$config_file" ]; then
  echo "  ✅ challenge_config.json found"
  python3 -c "
import json
with open('$config_file') as f:
    c = json.load(f)
print(f'     metric          : {c.get(\"metric\")}')
print(f'     use_probabilities: {c.get(\"use_probabilities\")}')
print(f'     daily_limit     : {c.get(\"daily_limit\")}')
print(f'     allowed_external: {c.get(\"allowed_external_data\")}')
" 2>/dev/null
else
  echo "  ❌ MISSING: $config_file"
fi

# ── SUMMARY ────────────────────────────────
echo ""
echo "========================================"
echo " AUDIT COMPLETE"
echo " Review ✅ / ⚠️  STUB / ❌ MISSING above"
echo " Paste full output back to Claude"
echo "========================================"
