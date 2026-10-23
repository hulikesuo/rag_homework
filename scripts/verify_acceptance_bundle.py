"""Offline verification of saved experiments. No generation API or model loads."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    commands = [
        ["scripts/check_t7_judge.py"],
        ["scripts/check_t3_subquery.py"],
        ["scripts/check_t3_retrieve_rrf.py"],
        ["scripts/check_demo_acceptance_stub_test.py"],
        ["scripts/main_acceptance_audit.py"],
        ["scripts/main_final_acceptance_audit.py"],
        ["scripts/check_submission_acceptance.py"],
    ]
    env = dict(os.environ, RAG_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONIOENCODING="utf-8")
    outcomes = []
    for args in commands:
        if not (ROOT / args[0]).is_file():
            outcomes.append({"command": args, "status": "FAIL", "error": "script missing"})
            continue
        print("RUN " + " ".join(args), flush=True)
        try:
            result = subprocess.run([sys.executable, "-I", "-B", *args], cwd=ROOT,
                                    env=env, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=180)
            outcome = {"command": args, "exit_code": result.returncode,
                       "status": "PASS" if result.returncode == 0 else "FAIL",
                       "output_tail": (result.stdout + result.stderr)[-6000:]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            outcome = {"command": args, "status": "FAIL", "error": str(exc)}
        outcomes.append(outcome)
        print(outcome["status"], flush=True)
    area = ROOT / "repair_outputs/acceptance_20261023"
    if not area.exists():
        area = ROOT / "results/acceptance_20261023"
    for name in ("main_new_results_audit.json", "main_final_artifact_audit.json"):
        path = area / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("errors") or data.get("pending"):
                outcomes.append({"command": [name], "status": "FAIL",
                                 "errors": data.get("errors"), "pending": data.get("pending")})
        else:
            outcomes.append({"command": [name], "status": "FAIL", "error": "audit missing"})
    delivery = area / "delivery"
    if not delivery.is_dir():
        delivery = ROOT / "delivery"
    check_report = delivery / "tests/check_report.json"
    if check_report.is_file():
        data = json.loads(check_report.read_text(encoding="utf-8"))
        if data.get("technical_status") != "PASS":
            outcomes.append({"command": ["check_report.json"], "status": "FAIL",
                             "error": "package inventory or submission technical checks failed"})
    else:
        outcomes.append({"command": ["check_report.json"], "status": "FAIL",
                         "error": "submission technical audit missing"})
    ok = all(item["status"] == "PASS" for item in outcomes)
    report = {"technical_status": "PASS" if ok else "FAIL", "course_status": "REQUIRES_USER_CONFIRMATION",
              "created_at": datetime.now(timezone.utc).isoformat(),
              "network_or_model_requested": False, "checks": outcomes,
              "note": "Writes audit JSON only; does not modify raw experimental rows. User student IDs, omitted division and human gold confirmation are outside this technical check."}
    out = area / "verification_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TECHNICAL " + report["technical_status"], flush=True)
    print("Report: " + str(out), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
