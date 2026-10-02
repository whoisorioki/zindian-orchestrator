import os
import warnings
from pathlib import Path
import pytest
import zindian.paths
from zindian.paths import CompetitionPaths

# Suppress known third-party noise that does not indicate test failures:
# 1. sklearn/LightGBM feature-name mismatch during eval-with-numpy-array
# 2. SHAP TreeExplainer list-of-ndarray behavior change for binary classifiers
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
    category=UserWarning,
)


def pytest_sessionstart(session):
    """Disable network access for tests by default.

    Set `ZINDIAN_ALLOW_NETWORK=1` in the environment to opt out
    when you intentionally want network-enabled tests.
    """
    if os.environ.get("ZINDIAN_ALLOW_NETWORK"):
        return
    os.environ["ZINDIAN_DISABLE_NETWORK"] = "1"


# Global tracking of the current test's tmp_path
_CURRENT_TMP_PATH = None


@pytest.fixture(autouse=True)
def store_tmp_path(request):
    global _CURRENT_TMP_PATH
    if "tmp_path" in request.fixturenames:
        _CURRENT_TMP_PATH = request.getfixturevalue("tmp_path")
    else:
        _CURRENT_TMP_PATH = None
    yield
    _CURRENT_TMP_PATH = None


# Store the original path resolver
_orig_resolve = zindian.paths.resolve_competition_paths

# The live competition tree. Tests must never write into it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_LIVE_COMPETITIONS = (_REPO_ROOT / "competitions").resolve()


def _assert_not_live(paths, context: str = "") -> None:
    """Fail loudly if a resolution points into the real competitions tree.

    A silent redirect would hide the next offender; the point is to make any
    write into the live tree a test failure.
    """
    comp_dir = getattr(paths, "competition_dir", None)
    if comp_dir is None:
        return
    try:
        resolved = Path(comp_dir).resolve()
    except Exception:  # pragma: no cover - defensive
        return
    if resolved == _LIVE_COMPETITIONS or _LIVE_COMPETITIONS in resolved.parents:
        raise RuntimeError(
            f"TEST SAFETY: {context or 'resolve_competition_paths'} resolved into "
            f"the LIVE competition tree ({resolved}). Tests must not read or "
            f"write real competition artifacts. Use tmp_path, or pass explicit "
            f"paths. Set ZINDIAN_TEST_ALLOW_LIVE=1 to override deliberately."
        )


@pytest.fixture(autouse=True)
def _forbid_live_competition_writes(request):
    """Autouse guard: fail any test that resolves into the live tree.

    Regression test for a real incident — the suite re-serialised
    scores/*.json, SKILL_STATE.json, reports/summaries/*, logs/* and
    experiments.db inside competitions/climate-risk-health-prediction-challenge
    on 2026-10-02, destroying evidence for the L1 investigation.
    """
    if os.environ.get("ZINDIAN_TEST_ALLOW_LIVE"):
        yield
        return

    resolved_paths = []

    def guarded(slug=None, **kwargs):
        paths = wrapped_resolve_competition_paths(slug, **kwargs)
        resolved_paths.append(paths)
        _assert_not_live(paths, f"resolve_competition_paths(slug={slug!r})")
        return paths

    zindian.paths.resolve_competition_paths = guarded
    try:
        yield
    finally:
        zindian.paths.resolve_competition_paths = wrapped_resolve_competition_paths


def wrapped_resolve_competition_paths(slug=None, **kwargs):
    global _CURRENT_TMP_PATH
    if _CURRENT_TMP_PATH is not None:
        root = _CURRENT_TMP_PATH
        selected_slug = slug
        if not selected_slug:
            env_slug = os.environ.get("COMPETITION_SLUG") or os.environ.get(
                "ZINDIAN_COMPETITION_SLUG"
            )
            if env_slug:
                if (root / "competitions" / env_slug).exists():
                    selected_slug = env_slug
                else:
                    # Ignore host env var if test has its own tmp_path competitions
                    tmp_matches = list(
                        (root / "competitions").glob("*/SKILL_STATE.json")
                    )
                    if not tmp_matches:
                        real_root = Path(__file__).resolve().parent.parent
                        if (real_root / "competitions" / env_slug).exists():
                            selected_slug = env_slug

        comp_dir = None
        if selected_slug:
            candidate = root / "competitions" / selected_slug
            if candidate.exists():
                comp_dir = candidate
            else:
                real_root = Path(__file__).resolve().parent.parent
                real_candidate = real_root / "competitions" / selected_slug
                if real_candidate.exists():
                    root = real_root
                    comp_dir = real_candidate
                else:
                    raise FileNotFoundError(
                        f"Competition '{selected_slug}' not found at {candidate} (or {real_candidate})."
                    )
        if comp_dir is None:
            matches = list((root / "competitions").glob("*/SKILL_STATE.json"))
            if len(matches) == 1:
                comp_dir = matches[0].parent
            elif len(matches) > 1:
                comp_dir = matches[0].parent
            else:
                real_root = Path(__file__).resolve().parent.parent
                real_matches = list(
                    (real_root / "competitions").glob("*/SKILL_STATE.json")
                )
                if len(real_matches) == 1:
                    root = real_root
                    comp_dir = real_matches[0].parent
                elif len(real_matches) > 1:
                    root = real_root
                    env_slug = os.environ.get("COMPETITION_SLUG")
                    if env_slug and (real_root / "competitions" / env_slug).exists():
                        comp_dir = real_root / "competitions" / env_slug
                    else:
                        comp_dir = real_matches[0].parent

        if comp_dir is None:
            return CompetitionPaths(
                root=root,
                competition_dir=None,
                state_path=root / "SKILL_STATE.json",
                config_path=root / "challenge_config.json",
                reports_dir=root / "reports",
                submissions_dir=root / "submissions",
                data_raw_dir=root / "data" / "raw",
                data_processed_dir=root / "data" / "processed",
                notebooks_dir=root / "notebooks",
            )
        return CompetitionPaths(
            root=root,
            competition_dir=comp_dir,
            state_path=comp_dir / "SKILL_STATE.json",
            config_path=comp_dir / "challenge_config.json",
            reports_dir=comp_dir / "reports",
            submissions_dir=comp_dir / "submissions",
            data_raw_dir=comp_dir / "data" / "raw",
            data_processed_dir=comp_dir / "data" / "processed",
            notebooks_dir=comp_dir / "notebooks",
        )
    else:
        return _orig_resolve(slug, **kwargs)


# Monkeypatch the paths module globally at import time for all tests
zindian.paths.resolve_competition_paths = wrapped_resolve_competition_paths
