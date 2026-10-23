"""One-command offline submission verification.

Run with -I -B so it never relies on installed project metadata:

  python -I -B scripts/verify_submission.py --quick

Quick mode runs the offline checkers that touch only local artifacts and
installed package metadata, then exits non-zero on any technical failure.
It never calls an API, never renders a PDF, and never writes formal results.
It does write isolated audit JSON under results/acceptance_20261023/audits/
(the acceptance sub-tree), leaving results/ top-level contents untouched.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(
        [sys.executable, "-I", "-B", *cmd],
        cwd=str(PROJECT_ROOT),
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run offline verify-only checkers (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="print the explicit costly commands; does not require approval",
    )
    args = parser.parse_args()

    if args.full:
        print("=== Costly opt-in commands (explicit, not run automatically) ===")
        print("python -I -B scripts/reproduce_acceptance.py --check")
        print("python -I -B scripts/main_final_acceptance_audit.py")
        print("python -I -B scripts/render_report_delivery.py --pdf")
        return 0

    commands = [
        [str(PROJECT_ROOT / "scripts/reproduce_acceptance.py"), "--check"],
        [str(PROJECT_ROOT / "scripts/main_acceptance_audit.py")],
        [str(PROJECT_ROOT / "scripts/main_final_acceptance_audit.py")],
        [str(PROJECT_ROOT / "scripts/check_submission_acceptance.py"), "--verify-only"],
        [str(PROJECT_ROOT / "scripts/check_build_package_safety.py")],
        [str(PROJECT_ROOT / "scripts/check_t5_delivery_figure.py")],
    ]
    results = [run(cmd) for cmd in commands]
    failures = [code for code in results if code != 0]
    print(f"verify-submission: ok={len(results) - len(failures)}/{len(results)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
