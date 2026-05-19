# Zindian Orchestrator — Agent Handoff v2.0
---

## ⚠️ Execution Directive 0 — Read This Before Anything Else

**The agent MUST NOT execute until `tabula init <competition_name>` has been run.**
The workspace, `.env`, and all 17 `SKILL.md` files must already exist before the
agent wakes up. If `SKILL_STATE.json` shows `dag_phase: "uninitialized"`, stop
immediately and tell the user to run `tabula init` first.

**Skills 13 (Ensembling) and 14 (Post-Processing) are Human-Gated.**
The agent MUST pause and ask: *"Ready to initiate Ensembling? (yes/no)"*
It MUST NOT run these skills autonomously under any circumstance.

---

## What This Project Is

An autonomous ML competition agent for Zindi Africa competitions. Every decision
the agent makes is conditional on the active competition's rules. No hardcoded
competition assumptions. The agent adapts to each challenge.

Every orchestrator action must declare which problem it serves: Problem 1 (generic Zindian agent) or Problem 2 (EY Biodiversity execution).

### EY-frogs Performance Snapshot

| Reference | OOF F1 | Threshold | LB |
|---|---:|---:|---:|
| Verified multi-seed blend | 0.84110 | 0.426 | 0.88350 |

---

## Environment

- Python 3.12.3 in virtualenv at `.venv/`
- Always activate: `source .venv/bin/activate`
- Packages: `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `shap`, `duckdb`, `PyGithub`, `python-dotenv`, `requests`
- Zindi CLI: `pip install git+https://github.com/eaedk/testing-zindi-package.git`
- Credentials: always loaded from `.env` via `python-dotenv` — never hardcoded

---

## Non-Negotiable Rules

1. Do not start until `tabula init` has populated the workspace.
2. Read `challenge_config.json` before touching any data.
3. Read and write `SKILL_STATE.json` at every state change.
4. Check `user.remaining_subimissions` before every Zindi submit call.
5. Lock MD5 hash of target column at Skill 01 — verify before every transform.
6. Submission comments must follow: `branch:X|oof_rmse:X|features:N|calib:X`
7. Use `git checkout -b <branch-name>` for every experiment — never experiment on `main`.
8. Feature engineering runs as 1 Anchor + 9 isolated variants — never stack untested features.
9. Gate every branch — only submit if OOF RMSE beats `anchor_oof_rmse` by ≥ 0.5%.
10. Skills 13 and 14 require explicit human `YES` before execution.
11. Select exactly 2 submissions for private judging — log rationale in `reports/`.
12. Never apply physical domain constraints unless `challenge_config.domain` confirms it.
15. Never use Latitude/Longitude as model features — banned per EY Biodiversity discussion 32369. Raw coords may only be used to extract TerraClimate values.
13. Never threshold predictions if `challenge_config.use_probabilities` is true.
14. Never use AutoML — almost always prohibited on Zindi.

---

## The 5 Kolesh Mechanics

### Mechanic 1 — tabula init Bootstrapper

The agent never builds workspace structure from scratch. A pre-agent CLI does it:

```bash
tabula init <competition_name>
```

This command does all of the following in one step:
- Clones the Zindian template repo into a new competition folder
- Copies all 17 `SKILL.md` files into `.opencode/skills/`
- Creates `SKILL_STATE.json` with `dag_phase: "phase_0_foundation"`
- Creates `challenge_config.json` with null skeleton
- Creates `.env` from `.env.example`
- Runs `git init` and makes the first commit: `"chore: tabula init <competition_name>"`

The agent only runs after this completes. This saves API tokens and eliminates
workspace setup as a point of failure.

### Mechanic 2 — Git Branching Per Experiment

Every experiment lives on its own git branch. No exceptions.

```bash
# When anchor baseline is confirmed (Skill 08)
git checkout -b anchor-baseline
git add -A && git commit -m "feat: anchor baseline | oof_rmse: X.XXXX"

# Before starting any feature experiment (Skill 07)
git checkout -b exp-feature-<name>

# If experiment fails the gate (Skill 11)
git checkout anchor-baseline      # instant reset — no line-by-line undo needed

# If experiment passes the gate — promote to new anchor
git checkout -b anchor-v2
git merge exp-feature-<name>
git add -A && git commit -m "feat: anchor v2 | oof_rmse: X.XXXX"
```

If an experiment corrupts the code, `git checkout anchor-baseline` resets everything
instantly. The agent never needs to undo edits manually.

### Mechanic 3 — 1 Anchor + 9 Variants Batching

Feature engineering never stacks untested changes. It runs in isolated rounds:

```
Round structure:
  anchor       → current best, already submitted and confirmed
  variant-01   → exactly ONE isolated change vs the anchor
  variant-02   → different isolated change vs the anchor
  ...
  variant-09   → ninth isolated change vs the anchor

After 9 variants:
  → Best passing variant becomes the new anchor
  → New round starts: new anchor + 9 new variants
```

The agent tracks the round in `SKILL_STATE.json`:

```json
{
  "feature_round": 1,
  "variants_tested": 3,
  "variants_passed": 1,
  "best_variant_this_round": "variant-02",
  "best_variant_oof_rmse": 0.3301
}
```

This prevents the classic agent failure mode of combining 10 untested features,
seeing a drop, and not knowing which feature caused it.

### Mechanic 4 — Human-Gated Skills (13 and 14)

When the agent reaches Skill 13 (Ensembling) or Skill 14 (Post-Processing),
it MUST output this and wait for input:

```
=== HUMAN GATE: Skill 13 — Ensembling ===
Current anchor OOF RMSE : X.XXXX
Variants passed this run : N
Feature rounds completed : N

Warning: Ensembling too early masks weak feature engineering.
Only proceed if you are satisfied with the feature quality.

Ready to initiate Ensembling? Type YES to continue or NO to return to Skill 07.
```

On `NO` → return to Skill 07 for another feature round.
On `YES` → set `human_gate_13_approved: true` in `SKILL_STATE.json` and proceed.

This prevents premature optimization of a weak feature foundation.

### Mechanic 5 — Authentication Resilience (Cookie Fallback)

Password auth sometimes fails due to anti-bot updates. The `.env` must include:

```
ZINDI_USERNAME=your_username
ZINDI_PASSWORD=your_password
ZINDI_COOKIE=your_browser_cookie_here
```

The submission wrapper tries auth in order:
1. Username + Password via Zindi CLI
2. On 401 Unauthorized → fall back to Cookie Auth via direct HTTP request
3. On both failing → halt, log the error, alert the user

To get your Zindi cookie: log in at zindi.africa → DevTools (`F12`) →
Application → Cookies → zindi.africa → copy the `_session` cookie value.

---

## Project Structure

```
zindian_orchestrator/
│
├── AGENTS.md                        ← master spec — ALL tools read this
├── CLAUDE.md                        ← copy for Claude Code
├── competitions/<slug>/SKILL_STATE.json  ← live DAG state and submission budget (per-competition)
├── competitions/<slug>/challenge_config.json ← competition rules (Skill 02 populates)
├── opencode.json                    ← OpenCode model config
├── .env                             ← credentials (never commit)
├── .gitignore
├── .venv/
│
├── .github/instructions/zindian.md  ← VS Code Copilot reads here
├── .cursor/rules/zindian.md         ← Cursor reads here
├── .windsurf/rules/zindian.md       ← Windsurf reads here
├── .kiro/specs/zindian.md           ← Kiro reads here
├── .opencode/agents/zindian.md      ← OpenCode reads here
│
├── specs/
│   ├── requirements.md              ← what the agent must do
│   ├── design.md                    ← how it does it
│   └── tasks.md                     ← current build checklist (source of truth)
│
├── tabula/
│   └── init.py                      ← tabula init CLI bootstrapper
│
├── zindian/                         ← Python agent package
│   ├── __init__.py
│   ├── state.py                     ← SKILL_STATE.json reader/writer
│   ├── config.py                    ← challenge_config.json reader with null guard
│   ├── ledger.py                    ← DuckDB experiment ledger
│   ├── zindi_client.py              ← Zindi wrapper + cookie fallback auth
│   └── skills/
│       ├── skill_01_integrity.py    ← MD5 hash lock
│       ├── skill_02_intake.py       ← competition rules parser
│       ├── skill_04_eda.py          ← violation EDA (conditional on domain)
│       ├── skill_05_cv.py           ← CV architect
│       ├── skill_06_cleaning.py     ← physical cleaning (conditional)
│       ├── skill_07_features.py     ← 1 anchor + 9 variants loop
│       ├── skill_08_anchor.py       ← baseline + git branch on confirm
│       ├── skill_09_calibration.py  ← group-level mean matching
│       ├── skill_10_shap.py         ← leakage detector + feature audit
│       ├── skill_11_gate.py         ← blocking gate + git checkout on fail
│       ├── skill_12_metric.py       ← metric trade-off analysis
│       ├── skill_13_fusion.py       ← HUMAN GATED — ensembling
│       ├── skill_14_inference.py    ← HUMAN GATED — post-processing
│       ├── skill_15_reporter.py     ← DuckDB + JSON + submission log
│       ├── skill_16_critique.py     ← self-critique slope audit (GO/NO_GO)
│       └── skill_17_governance.py   ← sub selection + reproducibility check
│
├── data/
│   ├── raw/                         ← downloaded once, never modified
│   └── processed/
│
├── notebooks/
│   ├── 01_integrity_audit.ipynb
│   ├── 02_challenge_intake.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_baseline.ipynb
│   ├── 05_features.ipynb
│   └── 06_calibration.ipynb
│
├── reports/
│   ├── experiments.json
│   ├── shap_analysis.json
│   └── submission_log.md
│
└── submissions/
    └── sub_001_anchor.csv
```

---

## SKILL_STATE.json Schema v2.0

```json
{
  "competition": null,
  "md5_target_hash": null,
  "current_git_branch": "main",
  "anchor_git_branch": null,
  "anchor_oof_rmse": null,
  "anchor_lb_score": null,
  "feature_round": 0,
  "variants_tested": 0,
  "variants_passed": 0,
  "best_variant_this_round": null,
  "best_variant_oof_rmse": null,
  "submissions_used_today": 0,
  "submissions_used_total": 0,
  "remaining_submissions": null,
  "dag_phase": "uninitialized",
  "human_gate_13_approved": false,
  "human_gate_14_approved": false,
  "selected_submissions": [],
  "last_updated": null
}
```

---

## challenge_config.json Schema

```json
{
  "name": null,
  "slug": null,
  "metric": null,
  "metric_direction": null,
  "submission_format": null,
  "use_probabilities": false,
  "daily_limit": null,
  "total_limit": null,
  "public_split_pct": null,
  "private_split_pct": null,
  "team_allowed": null,
  "code_review_tier": null,
  "allowed_external_data": false,
  "automl_permitted": false,
  "data_modality": null,
  "domain": null
}
```

---

## Git Branch Naming Convention

| Branch | When Created | Purpose |
|---|---|---|
| `main` | `tabula init` | Clean state — no experiments ever |
| `anchor-baseline` | After Skill 08 confirms first submission | First confirmed anchor |
| `anchor-v2`, `anchor-v3` | Each time anchor improves | Rolling best |
| `exp-feature-<name>` | Before each feature variant | Isolated feature test |
| `exp-calib-<name>` | Before calibration experiment | Isolated calibration test |
| `exp-ensemble-v1` | After human gate approved | Fusion experiment |

---

## Submission Budget

| Phase | Days | Max Subs/Day | Purpose |
|---|---|---|---|
| Anchor | 1–3 | 2 | Ground truth only |
| Exploration | 4 to N-7 | 5 | Gated variants only |
| Consolidation | Final 7 days | 3 | High-confidence ensembles only |
| Reserve | Always | 2 | Never fully exhaust daily limit |

---

## Zindi Client Setup (via ZindiClient wrapper)

The agent uses the `ZindiClient` wrapper (in `zindian/zindi_client.py`) for safe, 
budget-guarded submissions. Real Zindi API (verified via `inspect_zindi.py`):

### Real Zindi API Signatures
```python
# Zindi.__init__() signature:
Zindian(username, fixed_password=None)   # Only 2 params — NO agent-mode flags!

# Challenge selection (interactive only):
user.select_a_challenge()                # Shows menu picker — NO automation support

# Submit method:
user.submit(filepaths=[], comments=[])   # Takes lists, returns nothing

# Available properties:
user.my_rank                             # int | None
user.remaining_subimissions              # int | None  
user.which_challenge                     # str | None
user.username                            # str
```

### Using ZindiClient (recommended)
```python
from zindian.zindi_client import ZindiClient
from dotenv import load_dotenv

load_dotenv()

# Initialize with only username + password
client = ZindiClient.from_env()

# Check budget before submit
remaining = client.remaining_submissions_today()
print(f"Remaining subs today: {remaining}")

# Submit with structured comment (budget guard included)
result = client.submit(
    filepaths=["submission.csv"],
    branch="exp-feature-x",
    oof_rmse=0.2520,
    features=85,
    calib="none"
)

print(f"My rank: {result['my_rank']}")
print(f"Comment: {result['comment']}")
```

**Installation (critical — must be from KameniAlexNea fork):**

```bash
pip uninstall zindi -y
pip install git+https://github.com/KameniAlexNea/zindi.git
```

### Important Limitations
- ❌ Cannot select challenge via API (`challenge_id` parameter does NOT exist)
- ❌ No agent-mode flags (`return_models=True`, `to_print=False` do NOT exist)
- ❌ `submit()` returns nothing — cannot poll for rank immediately after
- ✅ Budget guard: Always check `remaining_subimissions` before submit
- ✅ Structured comment: Always use format `branch:X|oof_rmse:X|features:N|calib:X`

---

## What the Agent Must Never Do

- Start before `tabula init` has run
- Experiment directly on `main` or any anchor branch
- Stack multiple untested features in one variant
- Run Skill 13 or 14 without explicit human `YES`
- Submit without checking `remaining_subimissions`
- Commit `.env` to git
- Apply physical/solar constraints to non-solar competitions
- Use AutoML tools
- Use external data unless `challenge_config.allowed_external_data` is true
- Apply thresholding if `challenge_config.use_probabilities` is true
- Put reusable code in the repository root when an appropriate folder exists
- Introduce ad hoc file or folder names that break the repo's established naming conventions
- Hardcode dataset filenames or paths in skills when they should come from state, config, or workspace layout

---

## Session Start Prompt (Use This Every Time)

```
Read SKILL_STATE.json and challenge_config.json.
If dag_phase is "uninitialized" — stop and tell me to run tabula init.
Otherwise tell me:
  - Current phase
  - Current git branch
  - Anchor OOF RMSE
  - Submissions remaining today
  - Feature round and variants tested
  - Next unchecked task in specs/tasks.md
Then proceed in Plan mode only. Do not write any code yet.
```

When plan is approved:

```
Plan approved. Switch to Build mode.
After each file is written, update SKILL_STATE.json and run git add + git commit.
```