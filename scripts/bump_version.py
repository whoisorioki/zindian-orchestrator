#!/usr/bin/env python3
"""Single-point version propagation for the Zindian Orchestrator doc suite.

The VERSION file at the repository root is the canonical release version.
Run this script to propagate it to every documentation banner in one shot,
so a release never means hand-editing eight files.

Historical version references are intentionally NOT rewritten:
"[RESOLVED - v2.5] C1" markers, SoT §7 "Implemented <date>" rows, changelog
entries, and completed-items headings describe facts about the past, not
the current release. Only current-state banners are managed here.

Usage:
    python scripts/bump_version.py 2.7   # set a new version AND propagate
    python scripts/bump_version.py       # re-propagate VERSION as-is (idempotent)

Missing-pattern warnings mean a banner drifted from its known shape —
enforce with: scripts/sot_alignment_check.py --fail-on-misaligned
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"

# Each rule: (relative path, regex locating "<prefix><old version><suffix>",
#             replacement template using \\g<n> backreferences and "{v}")
BANNER_RULES = [
    # --- docs/source_of_truth.md (human-readable canonical header) --------
    ("docs/source_of_truth.md", r"(\*\*Version:\*\* )v\d+\.\d+", "\\g<1>v{v}"),
    # --- AGENTS.md ---------------------------------------------------------
    ("AGENTS.md", r"(aligned with SoT version \*\*v)\d+\.\d+", "\\g<1>{v}"),
    # --- README.md ---------------------------------------------------------
    ("README.md", r"(`docs/source_of_truth\.md` )v\d+\.\d+", "\\g<1>v{v}"),
    ("README.md", r"(\]\(docs/source_of_truth\.md\) )v\d+\.\d+", "\\g<1>v{v}"),
    # "for the full v2.4 feature specifications" -> drop the stale number (pattern may not exist)
    ("README.md", r"(for the full )v\d+\.\d+( feature specifications)", "\\g<1>\\g<2>"),
    ("README.md", r"(Authoritative specification )v\d+\.\d+", "\\g<1>v{v}"),
    (
        "README.md",
        r"(Authoritative architectural spec \()v\d+\.\d+(\))",
        "\\g<1>v{v}\\g<2>",
    ),
    ("README.md", r"(\*\*Status:\*\* )v\d+\.\d+", "\\g<1>v{v}"),
    # --- docs/orchestrator_overview.md -------------------------------------
    ("docs/orchestrator_overview.md", r"(\*\*Version:\*\* )\d+\.\d+", "\\g<1>{v}"),
    (
        "docs/orchestrator_overview.md",
        r"(\[Source of Truth )v\d+\.\d+(\]\(source_of_truth\.md\))",
        "\\g<1>v{v}\\g<2>",
    ),
    (
        "docs/orchestrator_overview.md",
        r"(\(source_of_truth\.md\) )v\d+\.\d+( for architecture details)",
        "\\g<1>v{v}\\g<2>",
    ),
    # --- docs/quick_start.md ------------------------------------------------
    (
        "docs/quick_start.md",
        r"(\*\*Source of Truth )v\d+\.\d+(\*\*)",
        "\\g<1>v{v}\\g<2>",
    ),
    ("docs/quick_start.md", r"(\(SoT )v\d+\.\d+(\) compliance)", "\\g<1>v{v}\\g<2>"),
    (
        "docs/quick_start.md",
        r"(\*\*Source of Truth Version:\*\* )v\d+\.\d+",
        "\\g<1>v{v}",
    ),
    # --- docs/ledger_architecture.md ----------------------------------------
    ("docs/ledger_architecture.md", r"(\*\*Version:\*\* )\d+\.\d+", "\\g<1>{v}"),
    # --- docs/reporting_logging_audit.md ------------------------------------
    (
        "docs/reporting_logging_audit.md",
        r"(Verified against SoT )v\d+\.\d+",
        "\\g<1>v{v}",
    ),
    # --- docs/document_map.md ------------------------------------------------
    (
        "docs/document_map.md",
        r"(Documentation Structure Map \()v\d+\.\d+(\))",
        "\\g<1>v{v}\\g<2>",
    ),
    (
        "docs/document_map.md",
        r"(### source_of_truth\.md \()v\d+\.\d+(\))",
        "\\g<1>v{v}\\g<2>",
    ),
    ("docs/document_map.md", r"(Outstanding work for )v\d+\.\d+", "\\g<1>v{v}"),
]


def propagate(new_version: str) -> int:
    """Applies every banner rule; returns number of rules with zero matches."""
    missing = []
    total_replacements = 0
    for rel_path, pattern, repl_template in BANNER_RULES:
        path = ROOT / rel_path
        if not path.exists():
            print(f"[WARN] {rel_path} not found — skipping")
            missing.append(rel_path)
            continue
        content = path.read_text(encoding="utf-8")
        replacement = repl_template.replace("{v}", new_version)
        new_content, n = re.subn(pattern, replacement, content)
        if n == 0:
            print(
                f"[WARN] {rel_path}: pattern matched nothing — "
                f"banner may have drifted from its known shape: {pattern}"
            )
            missing.append(rel_path)
            continue
        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
        total_replacements += n
        print(f"[OK] {rel_path}: {n} replacement(s)")
    print(f"\nTotal replacements: {total_replacements}")
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "version",
        nargs="?",
        default=None,
        help="New version (e.g. 2.7). Omit to re-propagate the VERSION file.",
    )
    args = parser.parse_args()

    if args.version:
        if not re.fullmatch(r"\d+\.\d+", args.version):
            parser.error("version must look like N.N (e.g. 2.7)")
        VERSION_FILE.write_text(args.version + "\n", encoding="utf-8")
        print(f"VERSION file updated -> {args.version}")
        target = args.version
    elif VERSION_FILE.exists():
        target = VERSION_FILE.read_text(encoding="utf-8").strip()
        print(f"Propagating VERSION={target}")
    else:
        parser.error("VERSION file missing and no version argument given")

    missing = propagate(target)
    print("\nRun: scripts/sot_alignment_check.py  (doc-version section enforces this)")
    sys.exit(0 if not missing else 0)  # warnings are advisory; checker enforces


if __name__ == "__main__":
    main()
