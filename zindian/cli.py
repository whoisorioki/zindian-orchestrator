#!/usr/bin/env python3
"""CLI for Zindian orchestrator commands."""

import sys
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Zindian CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a file to Zindi")
    submit_parser.add_argument("file", help="Path to submission file")

    # Submission board command
    subparsers.add_parser("submissions", help="Show submission board")

    # Leaderboard command
    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    lb_parser.add_argument("--per-page", type=int, default=20, help="Number of entries")

    # Ledger commands
    ledger_parser = subparsers.add_parser("ledger", help="Query experiments ledger")
    ledger_sub = ledger_parser.add_subparsers(dest="ledger_command")
    ledger_sub.add_parser("experiments", help="Show all experiments")
    ledger_sub.add_parser("submissions", help="Show all submissions")
    ledger_sub.add_parser("best", help="Show best experiment")
    ledger_sub.add_parser("passed", help="Show passed experiments")
    ledger_sub.add_parser("failed", help="Show failed experiments")

    # Monitor command
    subparsers.add_parser("monitor", help="Monitor Zindi competition")

    # Report command
    subparsers.add_parser("report", help="Generate phase summary report")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Run reproducibility audit")
    audit_parser.add_argument("--slug", help="Competition slug")

    # Status command
    subparsers.add_parser("status", help="Show current competition status")

    # Sync command
    subparsers.add_parser("sync", help="Sync state with git and Zindi")

    args = parser.parse_args()

    if args.command == "submit":
        from zindian.skills.skill_16_submit import run

        result = run(args.file)
        print(f"\nResult: {result}")

    elif args.command == "submissions":
        from zindian.skills.skill_16_submit import show_submission_board

        show_submission_board()

    elif args.command == "leaderboard":
        from zindian.skills.skill_16_submit import pull_leaderboard

        pull_leaderboard(per_page=args.per_page)

    elif args.command == "ledger":
        from zindian.ledger import Ledger

        with Ledger() as ledger:
            if args.ledger_command == "experiments":
                exps = ledger.query(
                    "SELECT * FROM experiments ORDER BY created_at DESC"
                )
                print(json.dumps(exps, indent=2, default=str))
            elif args.ledger_command == "submissions":
                subs = ledger.query(
                    "SELECT * FROM submissions ORDER BY submitted_at DESC"
                )
                print(json.dumps(subs, indent=2, default=str))
            elif args.ledger_command == "best":
                best = ledger.get_best_experiment()
                print(json.dumps(best, indent=2, default=str))
            elif args.ledger_command == "passed":
                passed = ledger.get_passed_experiments()
                print(json.dumps(passed, indent=2, default=str))
            elif args.ledger_command == "failed":
                failed = ledger.get_failed_experiments()
                print(json.dumps(failed, indent=2, default=str))
            else:
                ledger_parser.print_help()

    elif args.command == "monitor":
        from zindian.skills.skill_00_zindi_monitor import run
        from zindian.config import ChallengeConfig
        from zindian.state import SkillStateStore
        from zindian.paths import resolve_competition_paths

        paths = resolve_competition_paths()
        config = ChallengeConfig.load()
        state = SkillStateStore(paths.state_path).read()
        result = run(config, state)
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "report":
        from zindian.skills.skill_15_reporter import run

        result = run()
        print(json.dumps(result, indent=2, default=str))

    elif args.command == "audit":
        from zindian.skills.skill_22_reproducibility_audit import run

        result = run(slug=args.slug if hasattr(args, "slug") else None)
        sys.exit(0 if result.get("success") else 1)

    elif args.command == "status":
        from zindian.state import SkillStateStore
        from zindian.paths import resolve_competition_paths

        paths = resolve_competition_paths()
        state = SkillStateStore(paths.state_path).read()
        print(
            json.dumps(
                {
                    "competition": state.get("competition"),
                    "dag_phase": state.get("dag_phase"),
                    "submissions_used_today": state.get("submissions_used_today"),
                    "remaining_submissions": state.get("remaining_submissions"),
                    "anchor_oof_score": state.get("anchor_oof_score"),
                    "anchor_lb_score": state.get("anchor_lb_score"),
                    "current_git_branch": state.get("current_git_branch"),
                },
                indent=2,
            )
        )

    elif args.command == "sync":
        from zindian.sync_state import sync_all

        result = sync_all()
        print("✅ State synchronized")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
