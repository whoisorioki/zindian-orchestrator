#!/bin/bash
# SKILL_STATE Optimization - Deployment Checklist
# Ensures all future competitions automatically use optimized storage

echo "=== SKILL_STATE Optimization Deployment ==="
echo ""

# 1. Verify module exists
if [ -f "tabula/skill_state.py" ]; then
    echo "✓ tabula/skill_state.py exists"
else
    echo "✗ Missing tabula/skill_state.py"
    exit 1
fi

# 2. Check if any files still use old pattern
echo ""
echo "Checking for old json.load/dump patterns with SKILL_STATE..."
OLD_PATTERN=$(grep -r "json.load.*SKILL_STATE\|json.dump.*SKILL_STATE" --include="*.py" . 2>/dev/null | grep -v "skill_state" | wc -l)
if [ "$OLD_PATTERN" -gt 0 ]; then
    echo "⚠ Found $OLD_PATTERN files using old pattern:"
    grep -r "json.load.*SKILL_STATE\|json.dump.*SKILL_STATE" --include="*.py" . 2>/dev/null | grep -v "skill_state" | head -5
    echo ""
    echo "Run: python scripts/migrate_skill_state.py"
else
    echo "✓ No old patterns found"
fi

# 3. Verify all competitions have scores/ directory
echo ""
echo "Checking competitions..."
for comp in competitions/*/SKILL_STATE.json; do
    if [ -f "$comp" ]; then
        dir=$(dirname "$comp")
        if [ ! -d "$dir/scores" ]; then
            echo "⚠ $dir needs migration"
        fi
    fi
done

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "For new competitions, just use:"
echo "  from tabula.skill_state import read_state, write_state"
