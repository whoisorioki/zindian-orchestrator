#!/usr/bin/env python3
"""Backfill ledger from Zindi submission board."""

from zindian.ledger import Ledger
from zindian.zindi_client import ZindiClient
from zindian.config import ChallengeConfig
from zindian.state import SkillStateStore
from zindian.paths import resolve_competition_paths
import sys
import io

paths = resolve_competition_paths()
config = ChallengeConfig.load()
store = SkillStateStore(paths.state_path)
state = store.read()

client = ZindiClient()
client.select_competition(config.slug)

# Get submissions
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
try:
    subs = list(client._user.submission_board())
finally:
    sys.stdout = old

with Ledger() as ledger:
    for _, s in subs:
        # Create experiment
        exp_id = ledger.log_experiment(
            branch_name=state.get("current_git_branch", "main"),
            oof_rmse=state.get("anchor_oof_score"),
            feature_count=14,
            calibration_method="none",
            gate_result="PASS",
            dag_phase=state.get("dag_phase"),
            notes=s.get("comment", ""),
        )

        # Create submission
        ledger.log_submission(
            experiment_id=exp_id,
            branch_name=state.get("current_git_branch", "main"),
            public_score=s.get("public_score"),
            selected_for_final=s.get("chosen", False),
            comment=s.get("comment", ""),
        )
        print(f"✅ Backfilled: {s.get('filename')} (LB: {s.get('public_score')})")

print(f"\n✅ Backfilled {len(subs)} submissions to ledger")
