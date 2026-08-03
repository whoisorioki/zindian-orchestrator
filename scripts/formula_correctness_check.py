"""
Formula correctness checker for SoT <-> code alignment.

sot_alignment_check.py (already built) answers: "does the code contain a
symbol matching this S-item's name?" This script answers a different
question: "does the code's formula actually compute the same thing the
SoT's formula describes?" — which is what Findings A.1 and A.3 needed and
what no existing tool in this project checks.

Two independent techniques, both demonstrated below against real findings
from this project so their value is provable rather than asserted:

  1. NUMERIC EQUIVALENCE — evaluate the SoT's transcribed formula and the
     code's formula across many random parameter draws; if they don't agree
     to float tolerance across the whole domain, they are not the same
     formula, no matter how similar the code looks. Demonstrated on
     Finding A.1 (SE_OOF double-/K bug).

  2. DIMENSIONAL / UNIT INVARIANCE — for formulas that are supposed to be
     scale-invariant (like a normalized metric's gate margin), rescale the
     inputs' units and confirm the formula's *decision output* doesn't
     change. Demonstrated on Finding A.3 (MASE margin reintroducing target
     units into a supposedly dimensionless metric).

HOW TO ADAPT:
  - Keep a `SOT_FORMULAS` dict, one entry per S-item, transcribed by hand
    from the SoT prose/LaTeX whenever the SoT changes. This is intentionally
    manual — sympy's LaTeX parser is unreliable on hand-written markdown
    LaTeX, and a wrong auto-parse would be worse than no check at all.
  - Replace the CODE_FORMULAS stand-ins with real imports from the repo
    (skill_12_metric.py, skill_11_gate.py, etc).
  - Run this as a CI step alongside sot_alignment_check.py, not instead of
    it — they catch different failure classes.
"""

import math
import random
import sys
from pathlib import Path
from typing import Callable

# Ensure repo root is on sys.path so `zindian` imports resolve when this
# script is run directly from scripts/ (e.g. `python scripts/formula_correctness_check.py`).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Technique 1 — numeric equivalence over random domain sampling
# ---------------------------------------------------------------------------


def check_numeric_equivalence(
    name: str,
    sot_formula: Callable[..., float],
    code_formula: Callable[..., float],
    param_sampler: Callable[[], dict],
    n_trials: int = 2000,
    rel_tol: float = 1e-9,
) -> dict:
    """Draws n_trials random parameter sets, evaluates both formulas, and
    reports the first disagreement plus the overall pass rate. A single
    disagreement anywhere in the domain means the formulas are not
    equivalent — this is not a statistical test, it's an existence check
    for a counterexample."""
    mismatches = []
    for i in range(n_trials):
        params = param_sampler()
        sot_val = sot_formula(**params)
        code_val = code_formula(**params)
        if not math.isclose(sot_val, code_val, rel_tol=rel_tol, abs_tol=1e-12):
            mismatches.append((params, sot_val, code_val))
    result = {
        "name": name,
        "n_trials": n_trials,
        "n_mismatches": len(mismatches),
        "passed": len(mismatches) == 0,
        "first_mismatch": mismatches[0] if mismatches else None,
    }
    return result


# ---------------------------------------------------------------------------
# Technique 2 — unit-rescaling invariance check
# ---------------------------------------------------------------------------


def check_unit_invariance(
    name: str,
    decision_formula: Callable[..., bool],
    param_sampler: Callable[[], dict],
    unit_rescaled_keys: list[str],
    scale_factors: list[float] = [1.0, 1000.0, 0.001, 1e6],
    n_trials: int = 500,
) -> dict:
    """For a formula that produces a pass/fail decision and is claimed to be
    scale-invariant (e.g. because it operates on an already-normalized
    metric), rescale the named keys by each factor and confirm the decision
    doesn't flip. A decision that changes under a pure unit rescaling (e.g.
    kg -> g) means physical units have leaked into what should be a
    dimensionless comparison — exactly Finding A.3."""
    violations = []
    for i in range(n_trials):
        base_params = param_sampler()
        base_decision = decision_formula(**base_params)
        for scale in scale_factors:
            scaled_params = dict(base_params)
            for k in unit_rescaled_keys:
                scaled_params[k] = scaled_params[k] * scale
            scaled_decision = decision_formula(**scaled_params)
            if scaled_decision != base_decision:
                violations.append((base_params, scale, base_decision, scaled_decision))
                break
    result = {
        "name": name,
        "n_trials": n_trials,
        "n_violations": len(violations),
        "passed": len(violations) == 0,
        "first_violation": violations[0] if violations else None,
    }
    return result


# ---------------------------------------------------------------------------
# Demonstration 1 — Finding A.1 (SE_OOF double-/K bug), numeric equivalence
# ---------------------------------------------------------------------------


def _sample_nb_params():
    return dict(
        var_sample=random.uniform(0.01, 100.0),
        K=random.choice([3, 5, 10]),
        n_val=random.uniform(1, 50),
        n_train=random.uniform(51, 500),
    )


def sot_se_oof(var_sample, K, n_val, n_train):
    """Transcribed from SoT line 1537-1539: SE_OOF = sqrt(Var_NB), and
    Var_NB = Var_sample * (1/K + n_val/n_train)."""
    nb_factor = (1.0 / K) + (n_val / n_train)
    var_nb = var_sample * nb_factor
    return math.sqrt(var_nb)


def code_se_oof_current_shipped(var_sample, K, n_val, n_train):
    """Stand-in for the PRE-FIX code per the handoff doc:
    SE_OOF = sqrt(Var_NB / K) — the double-scaling bug. Kept only to
    demonstrate the checker catches it; the live repo no longer contains
    this bug (verified 2026-08-03)."""
    nb_factor = (1.0 / K) + (n_val / n_train)
    var_nb = var_sample * nb_factor
    return math.sqrt(var_nb / K)


def code_se_oof_after_fix(var_sample, K, n_val, n_train):
    """REAL PRODUCTION CODE (wired 2026-08-03): skill_12_metric computes
    SE_OOF = sqrt(Var_NB) with Var_NB = Var_sample * _get_nb_factor(K, fold_sizes).
    For the equal-fold case with fold_sizes provided, the NB factor is
    (1/K) + (n_val/n_train) per fold, matching the SoT formula exactly."""
    from zindian.skills.skill_12_metric import _get_nb_factor

    nb_factor = _get_nb_factor(K, [(n_train, n_val)] * K)
    var_nb = var_sample * nb_factor
    return math.sqrt(var_nb)


# ---------------------------------------------------------------------------
# Demonstration 2 — Finding A.3 (MASE margin unit leakage), invariance check
# ---------------------------------------------------------------------------


def _sample_mase_params():
    mase_score = random.uniform(0.5, 2.0)  # dimensionless, already normalized
    gate_margin = random.uniform(0.01, 0.1)  # dimensionless config constant
    mae_naive_baseline = random.uniform(1.0, 1000.0)  # HAS physical target units
    return dict(
        mase_score=mase_score,
        gate_margin=gate_margin,
        mae_naive_baseline=mae_naive_baseline,
    )


def decision_current_spec(mase_score, gate_margin, mae_naive_baseline):
    """Current SoT spec (pre-fix, per Finding A.3):
    effective_gate_margin = gate_margin * MAE_naive_baseline
    Promotion passes if mase_score < effective_gate_margin.
    MAE_naive_baseline carries physical target units, so this margin is NOT
    dimensionless even though mase_score is — the bug."""
    effective_margin = gate_margin * mae_naive_baseline
    return mase_score < effective_margin


def decision_corrected_spec(mase_score, gate_margin, mae_naive_baseline):
    """Proposed fix: treat MASE like RMSLE — raw gate_margin, no scaling by
    a unit-carrying baseline."""
    effective_margin = gate_margin
    return mase_score < effective_margin


# ---------------------------------------------------------------------------
# Run both demonstrations
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("DEMO 1 — numeric equivalence: SoT formula vs currently-shipped code")
    print("=" * 78)
    r = check_numeric_equivalence(
        "SE_OOF (current shipped, expected to FAIL)",
        sot_se_oof,
        code_se_oof_current_shipped,
        _sample_nb_params,
    )
    print(r)
    assert not r["passed"], "expected the current shipped formula to disagree with SoT"

    r2 = check_numeric_equivalence(
        "SE_OOF (post-fix, expected to PASS)",
        sot_se_oof,
        code_se_oof_after_fix,
        _sample_nb_params,
    )
    print(r2)
    assert r2["passed"], "post-fix formula should match SoT exactly"

    print()
    print("=" * 78)
    print("DEMO 2 — unit-rescaling invariance: MASE gate margin")
    print("=" * 78)
    r3 = check_unit_invariance(
        "MASE gate decision (current spec, expected to FAIL)",
        decision_current_spec,
        _sample_mase_params,
        unit_rescaled_keys=["mae_naive_baseline"],
    )
    print(r3)
    assert not r3[
        "passed"
    ], "expected current spec's decision to flip under unit rescaling"

    r4 = check_unit_invariance(
        "MASE gate decision (corrected spec, expected to PASS)",
        decision_corrected_spec,
        _sample_mase_params,
        unit_rescaled_keys=["mae_naive_baseline"],
    )
    print(r4)
    assert r4["passed"], "corrected spec's decision should be unit-invariant"

    print()
    print("All demonstrations behaved as predicted — both techniques catch")
    print("their target finding and clear once the corresponding fix lands.")
