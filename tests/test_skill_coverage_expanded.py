import importlib
import pytest

SKILL_EXPORTS = [
    ("zindian.skills.skill_00_zindi_monitor", "run"),
    ("zindian.skills.skill_00_discussion_monitor", "run"),
    ("zindian.skills.skill_01_integrity", "run"),
    ("zindian.skills.skill_02_intake", "run"),
    ("zindian.skills.skill_03_legality", "run"),
    ("zindian.skills.skill_04_eda", "run"),
    ("zindian.skills.skill_05_cv", "run"),
    ("zindian.skills.skill_06_preprocessing", "run"),
    ("zindian.skills.skill_07_features", "run"),
    ("zindian.skills.skill_08_anchor", "run"),
    ("zindian.skills.skill_09_calibration", "run"),
    ("zindian.skills.skill_10_shap", "run"),
    ("zindian.skills.skill_11_gate", "run"),
    ("zindian.skills.skill_12_metric", "run"),
    ("zindian.skills.skill_13_ensemble", "run"),
    ("zindian.skills.skill_13_oracle_fusion", "run"),
    ("zindian.skills.skill_14_inference", "run"),
    ("zindian.skills.skill_15_reporter", "run"),
    ("zindian.skills.skill_16_submit", "run"),
    ("zindian.skills.skill_17_governance", "run_governance"),
    ("zindian.skills.skill_18_librarian", "run_librarian"),
    ("zindian.skills.skill_19_code_miner", "run_code_miner"),
    ("zindian.skills.skill_20_scientist", "run_scientist"),
    ("zindian.skills.skill_21_pseudo_label", "run"),
    ("zindian.skills.skill_22_reproducibility_audit", "run"),
]


@pytest.mark.parametrize("module_name, export_name", SKILL_EXPORTS)
def test_skill_modules_export_callables(module_name: str, export_name: str) -> None:
    module = importlib.import_module(module_name)
    assert hasattr(module, export_name)
    assert callable(getattr(module, export_name))
