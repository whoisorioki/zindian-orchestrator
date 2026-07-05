Skill 06 — Data Preprocessing
=============================

Purpose
-------
- Impute missing values and drop constant columns while enforcing order: MNAR indicator creation first, MCAR median/mode fill second, constant dropping third.
- Avoid test set leakage by using training-derived imputation medians/modes.

Primary implementation
----------------------
- `zindian/skills/skill_06_preprocessing.py`

Current implementation notes
----------------------------
- Requires `X_train` and `X_test` dataframes to be present in state.
- Creates binary indicators `{col}_is_missing` for MNAR columns.
- Imputes MCAR columns using median (for numeric) or mode (for categorical) values computed from train set.
- Drops columns that are constant in BOTH train and test sets to prevent schema misalignment.
- Places cleaned matrices back into state under `X_train_clean` and `X_test_clean`.

Commands
--------
- Invoked via orchestrator during Phase 2A:
  python3 -m zindian.orchestrator --run_phase 2

Audit findings & resolution status
----------------------------------
- **Dependency on Nested eda**: [RESOLVED] Verified that `skill_04_eda.py` nests its data-quality lists (like `mnar_columns` and `mcar_columns`) inside the `"eda"` key in the skill state. `skill_06_preprocessing.py` successfully reads from this nested location, ensuring indicators are created and MCAR/MNAR imputation is correctly applied.


Outputs
-------
- Updates `SKILL_STATE.json` with keys: `X_train_clean`, `X_test_clean`, and `cleaning` dict containing metadata on indicators, medians, and constant columns dropped.
