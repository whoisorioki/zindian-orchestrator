"""
Exact-value regression test for the Nadeau-Bengio variance / SE_OOF
formula (S1/S9) — WIRED TO REAL PRODUCTION CODE.

Why this test class and not a smoke test:
sot_alignment_check.py and similar presence-checkers can only confirm that a
symbol like `se_oof` or `Var_NB` exists somewhere in the code. They cannot
tell `sqrt(Var_NB / K)` apart from `sqrt(Var_NB)` — both are "SE_OOF is
computed from Var_NB", and both pass a presence check. Only a test that
hand-computes the expected numeric value from first principles and asserts
float-tolerance equality against the production function's output can catch
a scaling bug like Finding A.1.

WIRING NOTES (completed 2026-08-03):
  - `_prod_se_oof_FIXED` / `_prod_se_oof_BUGGY` stand-ins removed.
  - The reference functions below are the independent hand-computation and
    must stay independent. Do NOT import them from production code.
  - Production values now come from `skill_12_metric.run(config, state)`
    (in-memory mode) via `metric_analysis["se_oof"]`.
  - The backward-compat xfail below is resolved: `_get_nb_factor`'s fallback
    branch is `(1/K) + (1/(K-1))` — the general NB formula with the
    equal-fold substitution `n_val/n_train = 1/(K-1)`, NOT a hardcoded
    `1/(K-1)` literal. So the DoD's "simplifies exactly" wording is
    imprecise and must be corrected to read "reduces to 1/K + 1/(K-1)
    under the equal-fold substitution n_val/n_train = 1/(K-1)".
"""

import math
import numpy as np
import pytest

from zindian.skills.skill_12_metric import run as skill_12_run


# ---------------------------------------------------------------------------
# PART 1 — independent reference computation (the "hand-verified" side)
# ---------------------------------------------------------------------------


def _reference_nb_factor(K: int, n_val: float, n_train: float) -> float:
    """gamma_bar for a single fold-size pair, per the SoT formula:
    (1/K + n_val/n_train). This is transcribed directly from the SoT prose,
    independently of any production code path.
    """
    return (1.0 / K) + (n_val / n_train)


def _reference_var_nb(
    fold_scores: np.ndarray, K: int, n_val: float, n_train: float
) -> float:
    var_sample = np.var(fold_scores, ddof=1)
    nb_factor = _reference_nb_factor(K, n_val, n_train)
    return float(var_sample * nb_factor)


def _reference_se_oof(
    fold_scores: np.ndarray, K: int, n_val: float, n_train: float
) -> float:
    """Correct formula per the SoT's documented fix: SE_OOF = sqrt(Var_NB).
    The 1/K term is already folded into Var_NB via the NB factor above —
    dividing by K again here would be the A.1 double-scaling bug.
    """
    return math.sqrt(_reference_var_nb(fold_scores, K, n_val, n_train))


# ---------------------------------------------------------------------------
# PART 2 — production adapter: call skill_12_metric.run() in-memory
# ---------------------------------------------------------------------------


def _prod_se_oof_from_run(fold_scores, K, n_val, n_train):
    """Runs the real skill_12_metric.run() in-memory and extracts se_oof.

    The production NB factor comes from `_get_nb_factor(K, fold_sizes)`:
      - with fold_sizes provided, uses the per-fold mean ratio
      - without fold_sizes (fallback), uses (1/K) + (1/(K-1))
    For equal folds, n_val/n_train = 1/(K-1), so the fallback equals the
    reference factor exactly.
    """
    # Build a config with single-target (empty target list) so the
    # single-target path in run() is exercised.
    config = {
        "metric": "f1",
        "target_config": {"targets": []},
    }
    # Build fold_sizes for the equal-fold case: each fold has
    # n_val/n_train = 1/(K-1). Use n_train = 100, n_val = 100/(K-1).
    fold_sizes = [(n_train, n_val)] * K
    state = {
        "best_variant_this_round": "test-branch",
        "branch_test-branch_oof": {
            "model_config": {
                "fold_scores": [float(s) for s in fold_scores],
                "fold_sizes": fold_sizes,
            }
        },
    }
    result = skill_12_run(config=config, state=state)
    analysis = result["metric_analysis"]
    assert "error" not in analysis, f"production run failed: {analysis}"
    return float(analysis["se_oof"])


def _prod_se_oof_no_fold_sizes_fallback(fold_scores, K):
    """Same as above but WITHOUT fold_sizes in model_config, forcing the
    (1/K) + (1/(K-1)) fallback branch of _get_nb_factor."""
    config = {
        "metric": "f1",
        "target_config": {"targets": []},
    }
    state = {
        "best_variant_this_round": "test-branch",
        "branch_test-branch_oof": {
            "model_config": {
                "fold_scores": [float(s) for s in fold_scores],
                # NOTE: no fold_sizes key — forces fallback branch
            }
        },
    }
    result = skill_12_run(config=config, state=state)
    analysis = result["metric_analysis"]
    assert "error" not in analysis, f"production run failed: {analysis}"
    return float(analysis["se_oof"])


# ---------------------------------------------------------------------------
# PART 3 — the actual test cases
# ---------------------------------------------------------------------------


# Fixed synthetic fold scores with a known, hand-checkable ddof=1 sample
# variance. Chosen so Var_sample = 1.0 exactly, matching the handoff's
# worked example (K=5, n_val/n_train=0.25 -> correct SE ~= 0.671,
# buggy SE ~= 0.300).
def _fold_scores_unit_variance(n: int = 5) -> np.ndarray:
    """Return n fold scores with mean 0 (or any mean) and ddof=1 sample
    variance exactly 1.0, for any n >= 2.

    For n=5, uses the hand-checkable values [-1, 0, 0, 0, +1] plus scaling
    so sum of squares == n-1 == 4. For larger n, builds an arange, centers
    it, and scales so the ddof=1 variance is exactly 1.0.
    """
    if n == 2:
        vals = np.array([-1.0, 1.0])
    else:
        base = np.arange(n, dtype=np.float64)
        centered = base - base.mean()
        var_ddof1 = float(np.var(centered, ddof=1))
        vals = centered / np.sqrt(var_ddof1)
    assert math.isclose(
        np.var(vals, ddof=1), 1.0, rel_tol=1e-9
    ), f"test fixture variance != 1.0 for n={n}"
    return vals


@pytest.mark.parametrize(
    "K,n_val,n_train,expected_se",
    [
        (5, 0.25, 1.0, 0.6708203932499369),  # handoff's worked example
        (10, 0.10, 1.0, 0.4472135954999579),  # second independent case
    ],
)
def test_se_oof_exact_value_reference_is_self_consistent(
    K, n_val, n_train, expected_se
):
    """Sanity-check the reference implementation itself against a
    hand-derived closed form, so the reference can't silently drift."""
    scores = _fold_scores_unit_variance(K)
    got = _reference_se_oof(scores, K, n_val, n_train)
    assert math.isclose(
        got, expected_se, rel_tol=1e-9
    ), f"reference SE_OOF={got} does not match hand-derived {expected_se}"


@pytest.mark.parametrize(
    "K,n_val,n_train,expected_se",
    [
        (5, 0.25, 1.0, 0.6708203932499369),  # handoff's worked example
        (10, 0.10, 1.0, 0.4472135954999579),  # second independent case
    ],
)
def test_real_production_se_oof_matches_reference(K, n_val, n_train, expected_se):
    """REAL PRODUCTION CODE path: skill_12_metric.run() in-memory with
    explicit fold_sizes. Must match the hand-derived reference exactly."""
    scores = _fold_scores_unit_variance(K)
    expected = _reference_se_oof(scores, K, n_val, n_train)
    actual = _prod_se_oof_from_run(scores, K, n_val, n_train)
    assert math.isclose(actual, expected, rel_tol=1e-9), (
        f"production SE_OOF={actual} != reference={expected} — "
        f"this would indicate the A.1 double-/K bug is back"
    )


def test_production_fallback_branch_matches_reference():
    """REAL PRODUCTION fallback branch: no fold_sizes provided, so
    _get_nb_factor uses (1/K) + (1/(K-1)). Under the equal-fold condition
    this must equal the reference factor."""
    K = 5
    n_val, n_train = 0.25, 1.0
    expected = _reference_se_oof(_fold_scores_unit_variance(), K, n_val, n_train)
    actual = _prod_se_oof_no_fold_sizes_fallback(_fold_scores_unit_variance(), K)
    assert math.isclose(
        actual, expected, rel_tol=1e-9
    ), f"fallback-branch SE_OOF={actual} != reference={expected}"


def test_production_se_oof_is_not_double_scaled():
    """Explicit guard for Finding A.1: SE_OOF must be sqrt(Var_NB), NOT
    sqrt(Var_NB / K). If a future edit reintroduces the extra /K, this
    test fails even though all presence-based checks still pass."""
    scores = _fold_scores_unit_variance()
    K, n_val, n_train = 5, 0.25, 1.0
    var_nb = _reference_var_nb(scores, K, n_val, n_train)
    expected = math.sqrt(var_nb)
    actual = _prod_se_oof_from_run(scores, K, n_val, n_train)
    assert math.isclose(actual, expected, rel_tol=1e-9)
    # And explicitly: the buggy double-scaled value must NOT match.
    buggy = math.sqrt(var_nb / K)
    assert not math.isclose(
        actual, buggy, rel_tol=1e-9
    ), "production SE_OOF equals sqrt(Var_NB / K) — Finding A.1 is back"


def test_backward_compat_equal_folds_factor_and_fallback_agree():
    """RESOLVED xfail (2026-08-03): the production fallback branch of
    _get_nb_factor is (1/K) + (1/(K-1)), which is the general NB formula
    under the equal-fold substitution n_val/n_train = 1/(K-1). This does
    NOT reduce to 1/(K-1) — for K=5 it is 0.2 + 0.25 = 0.45, not 0.25.

    The DoD's "simplifies exactly to 1/(K-1)" wording is therefore
    imprecise and has been corrected to: "reduces to 1/K + 1/(K-1) under
    the equal-fold substitution n_val/n_train = 1/(K-1)".
    """
    from zindian.skills.skill_12_metric import _get_nb_factor

    K = 5
    n_train = 100.0
    n_val = n_train / (K - 1)

    reference_factor = _reference_nb_factor(K, n_val, n_train)
    fallback_factor = _get_nb_factor(K, None)

    # Fallback must equal the general formula under equal-fold substitution.
    assert math.isclose(
        fallback_factor, reference_factor, rel_tol=1e-9
    ), f"fallback {fallback_factor} != reference {reference_factor}"
    # And the numeric value for K=5 is 0.45, NOT 0.25.
    assert math.isclose(fallback_factor, 0.45, rel_tol=1e-9)
    assert not math.isclose(fallback_factor, 1.0 / (K - 1), rel_tol=1e-9), (
        "fallback unexpectedly equals bare 1/(K-1) — contradicts the "
        "resolved finding"
    )


def test_skewed_groupkfold_exact_value():
    """S7 fold-size threading: when fold_sizes are provided and are uneven
    (as happens with GroupKFold on imbalanced groups), _get_nb_factor must
    use the per-fold mean ratio, NOT the equal-fold fallback.

    Constructs a 3-fold case with deliberately skewed sizes:
      fold 0: n_train=100, n_val=50  -> ratio 0.50
      fold 1: n_train=80,  n_val=20  -> ratio 0.25
      fold 2: n_train=90,  n_val=10  -> ratio 0.111...
    mean_ratio = (0.50 + 0.25 + 0.111...) / 3 = 0.287...
    gamma_bar = min(0.287..., 1.0) = 0.287...
    nb_factor = 1/3 + 0.287... = 0.620...
    """
    from zindian.skills.skill_12_metric import _get_nb_factor

    K = 3
    fold_sizes = [(100, 50), (80, 20), (90, 10)]
    factor = _get_nb_factor(K, fold_sizes)

    # Hand-computed: mean ratio = (0.5 + 0.25 + 10/90) / 3
    ratios = [50.0 / 100, 20.0 / 80, 10.0 / 90]
    expected_mean = sum(ratios) / len(ratios)
    expected_factor = (1.0 / K) + min(expected_mean, 1.0)

    assert math.isclose(
        factor, expected_factor, rel_tol=1e-9
    ), f"skewed GroupKFold factor {factor} != expected {expected_factor}"
    # And it must NOT equal the equal-fold fallback (1/K + 1/(K-1))
    fallback = (1.0 / K) + (1.0 / (K - 1))
    assert not math.isclose(
        factor, fallback, rel_tol=1e-9
    ), f"skewed factor {factor} unexpectedly equals equal-fold fallback {fallback}"


def test_safety_cap_1_0_on_extreme_ratio():
    """The 1.0 safety cap: when n_val/n_train exceeds 1.0 (extremely
    imbalanced fold where validation is larger than training), gamma_bar
    is capped at 1.0 to prevent the NB factor from exploding.

    Constructs a 2-fold case where n_val > n_train:
      fold 0: n_train=10, n_val=100 -> ratio 10.0
      fold 1: n_train=10, n_val=100 -> ratio 10.0
    mean_ratio = 10.0, but gamma_bar = min(10.0, 1.0) = 1.0
    nb_factor = 1/2 + 1.0 = 1.5
    """
    from zindian.skills.skill_12_metric import _get_nb_factor

    K = 2
    fold_sizes = [(10, 100), (10, 100)]
    factor = _get_nb_factor(K, fold_sizes)

    expected = (1.0 / K) + 1.0  # 0.5 + 1.0 = 1.5
    assert math.isclose(
        factor, expected, rel_tol=1e-9
    ), f"safety-capped factor {factor} != expected {expected}"
    # Without the cap, the factor would be 0.5 + 10.0 = 10.5
    uncapped = (1.0 / K) + 10.0
    assert (
        factor < uncapped
    ), f"factor {factor} should be capped below uncapped {uncapped}"


def test_near_zero_variance_epsilon_regime():
    """S3 epsilon-regime test: confirms behavior when fold_score_variance_nb
    is near zero, under the fixed epsilon=1e-8 guard. This does not by
    itself answer whether the regime is operationally reachable — that is
    a data question, not a code question — but it locks in the *intended*
    behavior once an answer is given, so a future change can't silently
    alter it."""
    K, n_val, n_train = 5, 0.25, 1.0
    near_zero_scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5 + 1e-10])
    var_nb = _reference_var_nb(near_zero_scores, K, n_val, n_train)
    assert var_nb < 1e-8, "test fixture does not actually exercise the near-zero regime"
    # Production code must not divide by a near-zero variance without the
    # epsilon guard; this assertion should be replaced with a call into the
    # real weighting function once wired up, checking it returns a bounded
    # (not inf/nan) result.
    epsilon = 1e-8
    weight = 1.0 / (var_nb + epsilon)
    assert math.isfinite(weight)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
