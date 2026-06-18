# Zindian CLI Quick Reference

## Setup
```bash
cd /home/sagemaker-user/shared/zindian-orchestrator
export COMPETITION_SLUG="june-study-jam-series-transaction-volume-forecasting-challenge"
source .venv/bin/activate  # Activate virtual environment
python -m zindian.cli <command>
```

**Or use without activating:**
```bash
export COMPETITION_SLUG="june-study-jam-series-transaction-volume-forecasting-challenge"
.venv/bin/python -m zindian.cli <command>
```

## Core Commands

### status - Current Competition State
```bash
python -m zindian.cli status
```
**Returns:**
```json
{
  "competition": "<slug>",
  "dag_phase": "phase_3_anchor_promoted",
  "submissions_used_today": 10,
  "remaining_submissions": 8,
  "anchor_oof_score": 0.5545387402931464,
  "anchor_lb_score": 0.552117936,
  "current_git_branch": "anchor-v2"
}
```
**Fields:**
- `dag_phase` - Current pipeline phase
- `submissions_used_today` - Total submissions made
- `remaining_submissions` - Budget remaining
- `anchor_oof_score` - Out-of-fold validation score
- `anchor_lb_score` - Public leaderboard score
- `current_git_branch` - Active git branch

### sync - Update State from Git + Zindi
```bash
python -m zindian.cli sync
```
**Updates:**
- Current git branch → `SKILL_STATE["current_git_branch"]`
- Selected submissions → `SKILL_STATE["selected_submissions"]`
- Best LB score → `SKILL_STATE["anchor_lb_score"]`
- Current rank → `SKILL_STATE["anchor_rank"]`
- Remaining submissions → `SKILL_STATE["remaining_submissions"]`

### submissions - View Submission History
```bash
python -m zindian.cli submissions
```
**Output:** Formatted table with ID, date, LB score, chosen status, filename, comment

### leaderboard - View Competition Leaderboard
```bash
python -m zindian.cli leaderboard [--per-page N]
```
**Shows:** Current rank, top N competitors

### submit - Submit to Zindi
```bash
python -m zindian.cli submit submissions/sub_010_anchor.csv
```
**Workflow:**
1. Validates submission format
2. Checks budget (aborts if 0 remaining)
3. Human gate confirmation
4. Submits to Zindi
5. Records to ledger DB
6. Updates state

## Experiments Tracking

### ledger experiments - All Experiments
```bash
python -m zindian.cli ledger experiments
```
**Returns:** JSON array of all experiments with branch, OOF score, features, calibration, gate result

### ledger submissions - All Submissions
```bash
python -m zindian.cli ledger submissions
```
**Returns:** JSON array of all submissions with experiment_id, branch, public score, rank

### ledger best - Best Experiment
```bash
python -m zindian.cli ledger best
```
**Returns:** Single experiment with best score per `config["metric"]` and `config["metric_direction"]` (minimize for RMSE/RMSLE, maximize for F1/AUC)

### ledger passed/failed - Gate Results
```bash
python -m zindian.cli ledger passed
python -m zindian.cli ledger failed
```
**Returns:** Experiments filtered by gate_result

## Monitoring & Reporting

### monitor - Competition Monitoring
```bash
python -m zindian.cli monitor
```
**Checks:** Discussion board, data patches, competition status  
**Writes:** Logs community signals to `SKILL_STATE["community_signals"]` (config remains frozen post-Phase 1)

### report - Generate Phase Report
```bash
python -m zindian.cli report
```
**Creates:** `reports/phase_1_summary.json` with config, state, ledger stats

### audit - Reproducibility Audit
```bash
python -m zindian.cli audit [--slug competition-slug]
```
**Validates:**
- Requirements lockfile sync
- No AutoML imports
- Git branch alignment
- OOF cv_strategy_id tags

## Integration Points

### State File
**Read-only commands:** status, submissions, leaderboard, ledger  
**Write commands:** sync, submit, report, audit, monitor

- `status` - Reads `SKILL_STATE.json` (no writes)
- `sync` - Writes to `SKILL_STATE.json` (git branch, submissions, rank, LB score)
- `submit` - Writes to `SKILL_STATE.json` (submission count, budget, rank) + ledger DB
- `report` - Writes to `SKILL_STATE.json` (last_reported timestamp)
- `monitor` - Writes community_signals to `SKILL_STATE.json` (NOT challenge_config.json per Phase 1 freeze)
- `audit` - Writes reproducibility_audit to `SKILL_STATE.json`

### Ledger DB
Submissions auto-record to `competitions/<slug>/reports/experiments.db`

### Config
All commands respect `competitions/<slug>/challenge_config.json`

## For Developers

### Adding New Commands
1. Add subparser in `zindian/cli.py` (line 10-42)
2. Implement handler in elif block (line 48-145)
3. Update this reference

### Command Pattern
```python
if args.command == "mycommand":
    from zindian.module import function
    result = function()
    print(json.dumps(result, indent=2))
```

### Testing
```bash
pytest tests/test_submission_board_leaderboard_integration.py -v
```
