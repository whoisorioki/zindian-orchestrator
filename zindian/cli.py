#!/usr/bin/env python3
"""CLI for Zindian orchestrator commands.

Consolidated 21 commands covering the Competition Data Lifecycle.
"""

import sys
import argparse
import json
import subprocess
from pathlib import Path


def _run_shell_script(script_path: str, args: list[str] | None = None) -> None:
    """Execute a .sh script via bash/sh on Unix, or print error/instructions on Windows."""
    root = Path(__file__).resolve().parent.parent
    full_path = root / script_path
    
    if not full_path.exists():
        print(f"Error: Script not found at {full_path}")
        sys.exit(1)

    cmd_args = args or []

    if sys.platform == "win32":
        shells = ["bash", "sh"]
        for shell in shells:
            try:
                subprocess.run([shell, "--version"], capture_output=True, check=True)
                cmd = [shell, str(full_path)] + cmd_args
                result = subprocess.run(cmd, cwd=str(root))
                if result.returncode != 0:
                    sys.exit(result.returncode)
                return
            except (subprocess.SubprocessError, FileNotFoundError):
                continue
        
        print(f"\nError: A shell environment (bash/sh) is required to run this script.")
        print(f"To run this command on Windows, please execute it within Git Bash, WSL, or MSYS:")
        print(f"  bash {script_path} " + " ".join(cmd_args))
        sys.exit(1)
    else:
        shells = ["bash", "sh"]
        last_error = None
        for shell in shells:
            try:
                cmd = [shell, str(full_path)] + cmd_args
                result = subprocess.run(cmd, cwd=str(root))
                if result.returncode != 0:
                    sys.exit(result.returncode)
                return
            except Exception as e:
                last_error = e
        print(f"Error executing shell script: {last_error}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Zindian CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---------------------------------------------------------
    # Core Commands
    # ---------------------------------------------------------
    submit_parser = subparsers.add_parser("submit", help="Submit a file to Zindi")
    submit_parser.add_argument("file", help="Path to submission file")

    subparsers.add_parser("submissions", help="Show submission board")

    lb_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    lb_parser.add_argument("--per-page", type=int, default=20, help="Number of entries")

    ledger_parser = subparsers.add_parser("ledger", help="Query experiments ledger")
    ledger_sub = ledger_parser.add_subparsers(dest="ledger_command")
    ledger_sub.add_parser("experiments", help="Show all experiments")
    ledger_sub.add_parser("submissions", help="Show all submissions")
    ledger_sub.add_parser("best", help="Show best experiment")
    ledger_sub.add_parser("passed", help="Show passed experiments")
    ledger_sub.add_parser("failed", help="Show failed experiments")

    subparsers.add_parser("monitor", help="Monitor Zindi competition")
    subparsers.add_parser("report", help="Generate phase summary report")

    audit_parser = subparsers.add_parser("audit", help="Run reproducibility audit")
    audit_parser.add_argument("--slug", help="Competition slug")

    subparsers.add_parser("status", help="Show current competition status")
    subparsers.add_parser("sync", help="Sync state with git and Zindi")

    phase_parser = subparsers.add_parser("phase", help="Execute pipeline phase")
    phase_parser.add_argument(
        "phase_id",
        choices=["1", "2A", "2B", "3A", "3B", "4"],
        help="Phase to execute (1, 2A, 2B, 3A, 3B, 4)",
    )
    phase_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed skill output"
    )
    phase_parser.add_argument("--variant", help="Variant name to pass to phase skills")

    # ---------------------------------------------------------
    # Utility Commands
    # ---------------------------------------------------------
    bootstrap_parser = subparsers.add_parser("bootstrap", help="Bootstrap a new competition folder")
    bootstrap_parser.add_argument("slug", help="Competition slug (e.g. ey-frogs)")
    bootstrap_parser.add_argument("--move-files", action="store_true", help="Move detected root files to data/raw/")
    bootstrap_parser.add_argument("--yes", "-y", action="store_true", help="Assume yes for confirmations")

    subparsers.add_parser("init-ledger", help="Initialize the experiments ledger database")

    preflight_parser = subparsers.add_parser("preflight", help="Run preflight compliance verification checks")
    preflight_parser.add_argument("--competition", help="Competition folder path (e.g., competitions/ey-frogs)")

    subparsers.add_parser("preflight-sim", help="Run preflight simulation checks (tmpcomp and ey-frogs)")
    subparsers.add_parser("verify-state", help="Verify the competition state files and datasets")
    subparsers.add_parser("verify-phase-b", help="Verify Phase B package hardening assertions")
    subparsers.add_parser("write-oof-meta", help="Write per-OOF metadata JSON files alongside OOF CSVs")
    subparsers.add_parser("compile-requirements", help="Compile pinned requirements.txt from requirements.in")

    archive_parser = subparsers.add_parser("archive", help="Archive a completed competition folder (excludes CSVs)")
    archive_parser.add_argument("slug", help="Competition slug (e.g. ey-frogs)")

    subparsers.add_parser("audit-framework", help="Perform full framework audit of workspace, stubs, and venv")
    subparsers.add_parser("check-deployment", help="Check SKILL_STATE storage optimization and migration status")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Core Command Handlers
    # ---------------------------------------------------------
    if args.command == "submit":
        try:
            from zindian.skills.skill_16_submit import run as _submit_run
            result = _submit_run(args.file)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "submissions":
        try:
            from zindian.skills.skill_16_submit import show_submission_board
            show_submission_board()
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "leaderboard":
        try:
            from zindian.skills.skill_16_submit import pull_leaderboard
            pull_leaderboard(per_page=args.per_page)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "ledger":
        try:
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
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "monitor":
        try:
            from zindian.skills.skill_00_zindi_monitor import run as _monitor_run
            from zindian.config import ChallengeConfig
            from zindian.state import SkillStateStore
            from zindian.paths import resolve_competition_paths

            paths = resolve_competition_paths(require_competition=True)
            config = ChallengeConfig.load()
            state = SkillStateStore(paths.state_path).read()
            result = _monitor_run(config._data, state)
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "report":
        try:
            from zindian.skills.skill_15_reporter import run as _report_run
            result = _report_run()
            print(json.dumps(result, indent=2, default=str))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "audit":
        try:
            from zindian.skills.skill_22_reproducibility_audit import run as _audit_run
            result = _audit_run(slug=args.slug if hasattr(args, "slug") else None)
            if not result.get("success"):
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "status":
        try:
            from zindian.state import SkillStateStore
            from zindian.paths import resolve_competition_paths

            paths = resolve_competition_paths(require_competition=True)
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
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "sync":
        try:
            from zindian.sync_state import sync_all
            result = sync_all()
            print("State synchronized")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "phase":
        try:
            from zindian.orchestrator import run_phase

            print(f"\n{'=' * 60}")
            print(f"EXECUTING PHASE {args.phase_id}")
            print(f"{'=' * 60}\n")

            results = run_phase(args.phase_id, variant_name=getattr(args, "variant", None))

            print(f"\n{'=' * 60}")
            print(f"PHASE {args.phase_id} RESULTS")
            print(f"{'=' * 60}")

            all_success = True
            for skill_name, result in results.items():
                status = result.get("status", "UNKNOWN")
                print(f"\n{skill_name}: {status}")

                if status == "ERROR":
                    all_success = False
                    print(f"  Error: {result.get('message', 'Unknown error')}")
                    if args.verbose and "traceback" in result:
                        print(f"\n{result['traceback']}")
                elif status == "GO" and args.verbose:
                    print(f"  {result.get('message', '')}")

            if not all_success:
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # ---------------------------------------------------------
    # Utility Command Handlers
    # ---------------------------------------------------------
    elif args.command == "bootstrap":
        try:
            from scripts.bootstrap_competition import main as _bootstrap_main
            argv = [args.slug]
            if args.move_files:
                argv.append("--move-files")
            if args.yes:
                argv.append("--yes")
            _bootstrap_main(argv)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "init-ledger":
        try:
            root = Path(__file__).resolve().parent.parent
            result = subprocess.run([sys.executable, str(root / "scripts" / "init_ledger.py")], cwd=str(root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "preflight":
        try:
            root = Path(__file__).resolve().parent.parent
            cmd = [sys.executable, str(root / "scripts" / "preflight_enforce.py")]
            if args.competition:
                cmd.extend(["--competition", args.competition])
            result = subprocess.run(cmd, cwd=str(root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "preflight-sim":
        _run_shell_script("scripts/run_preflight_sim.sh")

    elif args.command == "verify-state":
        try:
            root = Path(__file__).resolve().parent.parent
            result = subprocess.run([sys.executable, str(root / "scripts" / "verify_competition_state.py")], cwd=str(root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "verify-phase-b":
        try:
            root = Path(__file__).resolve().parent.parent
            result = subprocess.run([sys.executable, str(root / "scripts" / "verify_phase_b.py")], cwd=str(root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "write-oof-meta":
        try:
            root = Path(__file__).resolve().parent.parent
            result = subprocess.run([sys.executable, str(root / "scripts" / "write_oof_meta.py")], cwd=str(root))
            if result.returncode != 0:
                sys.exit(result.returncode)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "compile-requirements":
        _run_shell_script("scripts/compile_requirements.sh")

    elif args.command == "archive":
        _run_shell_script("scripts/archive_competition.sh", [args.slug])

    elif args.command == "audit-framework":
        _run_shell_script("scripts/zindian_audit.sh")

    elif args.command == "check-deployment":
        _run_shell_script("scripts/check_skill_state_deployment.sh")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
