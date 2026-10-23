"""Fail-closed technical checks for the RAG course submission.

Run directly, without installing the project as a package:

  python -I -B scripts/check_submission_acceptance.py

The checker is read-only for formal data. Its only write is the machine-readable
report at delivery/tests/check_report.json. Technical PASS does not mean that
the course submission is accepted: unresolved human and experimental items are
reported separately as REQUIRED_HUMAN, and the global status stays
NOT_ACCEPTED.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DELIVERY_REL = Path("repair_outputs") / "acceptance_20261023" / "delivery"
_root_delivery = PROJECT_ROOT / "delivery"
DELIVERY = PROJECT_ROOT / DELIVERY_REL
if not DELIVERY.is_dir() and _root_delivery.is_dir():
    DELIVERY = _root_delivery
RESULTS = PROJECT_ROOT / "results"
QA_SET = PROJECT_ROOT / "data" / "eval" / "qa_set.jsonl"
SNAPSHOT = DELIVERY / "formal_results_snapshot.json"
LOCK_FILE = DELIVERY / "requirements-lock.txt"
LOCK_STATUS = DELIVERY / "requirements-lock-status.json"
CHECK_REPORT = DELIVERY / "tests" / "check_report.json"

OUTPUT_NAME = "\u5b66\u53f7_RAG\u4f5c\u4e1a_\u9a8c\u6536\u4fee\u8ba2\u7248_20261023.zip"
QA_SHA256 = "bfcbefc1ba2519c9728451559d30fd55f05151f903a927d836011ba07ae7b969"
EXPECTED_LEVELS = {"L1": 8, "L2": 8, "L3": 7, "L4": 7}
EXPECTED_RESULT_COUNT = 42
MAX_PACKAGE_FILE_BYTES = 50 * 1024 * 1024

T4_EXPECTED = {
    "bm25": {"Hit@5": "0.8333", "Recall@5": "0.7889", "MRR@10": "0.7093"},
    "dense": {"Hit@5": "0.8333", "Recall@5": "0.6972", "MRR@10": "0.6469"},
    "hybrid": {"Hit@5": "0.8667", "Recall@5": "0.7972", "MRR@10": "0.7290"},
}

REQUIRED_T1_EVIDENCE_IDS = {"q06", "q07", "q22", "q28"}

CANDIDATE_ROOT_FILES = ("README.md", "\u62a5\u544a.md", "requirements.txt", ".env.example")
SOURCE_FILES = (
    "src/__init__.py",
    "src/advanced_chunking.py",
    "src/chains.py",
    "src/chunker.py",
    "src/compression.py",
    "src/config.py",
    "src/embed.py",
    "src/evaluate.py",
    "src/generator.py",
    "src/index_store.py",
    "src/loader.py",
    "src/pipeline.py",
    "src/rerank.py",
    "src/retriever.py",
    "src/subquery.py",
    "src/textutil.py",
    "src/utils.py",
)
EXPERIMENT_FILES = (
    "experiments/__init__.py",
    "experiments/common.py",
    "experiments/run_all.py",
    "experiments/t1_baseline.py",
    "experiments/t2_advanced_chunking.py",
    "experiments/t2_chunking.py",
    "experiments/t3_query_opt.py",
    "experiments/t4_retriever_compare.py",
    "experiments/t5_rerank_compress.py",
    "experiments/t6_advanced.py",
    "experiments/t7_eval.py",
)
DEMO_FILES = ("demo/__init__.py", "demo/app.py")
SCRIPT_FILES = (
    "scripts/__init__.py",
    "scripts/annotate_llm.py",
    "scripts/build_case_evidence.py",
    "scripts/build_qaset.py",
    "scripts/build_acceptance_package.py",
    "scripts/check_build_package_safety.py",
    "scripts/check_case_evidence.py",
    "scripts/check_compression.py",
    "scripts/check_handover_issues.py",
    "scripts/check_restoration.py",
    "scripts/check_submission_acceptance.py",
    "scripts/check_submission_results.py",
    "scripts/check_t2_advanced.py",
    "scripts/check_t3_retrieve_rrf.py",
    "scripts/check_t3_subquery.py",
    "scripts/check_t5_delivery_figure.py",
    "scripts/check_t7_judge.py",
    "scripts/check_demo_acceptance.py",
    "scripts/check_demo_acceptance_stub_test.py",
    "scripts/check_t3_subquery.py",
    "scripts/check_t3_retrieve_rrf.py",
    "scripts/audit_t7_worst5_provenance.py",
    "scripts/llm_judge_t7.py",
    "scripts/main_acceptance_audit.py",
    "scripts/main_final_acceptance_audit.py",
    "scripts/make_figures.py",
    "scripts/pick_qaset.py",
    "scripts/prepare_submission_results.py",
    "scripts/qa_annotation.csv",
    "scripts/render_report_delivery.py",
    "scripts/reproduce_acceptance.py",
    "scripts/restore_submission_results.py",
    "scripts/run_t4_acceptance.py",
    "scripts/run_t4_supplement.py",
    "scripts/t4_acceptance_queries.jsonl",
    "scripts/t4_supplement_queries.jsonl",
    "scripts/t4_cases.py",
    "scripts/t5_delivery_figure.py",
    "scripts/verify_submission.py",
    "scripts/verify_acceptance_bundle.py",
    "scripts/write_report_architecture_png.py",
    "scripts/write_delivery_requirements_lock.py",
)
GENERATED_DELIVERY_NAMES = {"manifest.json", "SHA256SUMS.txt"}


class Check:
    def __init__(self, check_id: str, description: str, ok: bool, detail: str, hard: bool = False):
        self.check_id = check_id
        self.description = description
        self.ok = bool(ok)
        self.detail = detail
        self.hard = bool(hard)

    def as_dict(self) -> dict:
        return {
            "id": self.check_id,
            "description": self.description,
            "status": "PASS" if self.ok else "FAIL",
            "detail": self.detail,
            "fail_closed": self.hard,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path.name}:{line_number} is not a JSON object")
                rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _decimal_equal(actual: str, expected: str) -> bool:
    try:
        return Decimal(actual.strip()) == Decimal(expected)
    except (AttributeError, InvalidOperation):
        return False


def _add(checks: list[Check], check_id: str, description: str, ok: bool, detail: str, hard: bool = False):
    checks.append(Check(check_id, description, ok, detail, hard))


def _forbidden_reason(rel: Path) -> str | None:
    parts = rel.parts
    name = rel.name
    lower_name = name.lower()
    if ".git" in parts:
        return "git metadata"
    if "_cache" in parts:
        return "cache directory"
    if "__pycache__" in parts:
        return "Python bytecode cache"
    if ".tmp_docx" in parts:
        return "temporary DOCX extraction"
    if name == ".env":
        return "secret environment file"
    if lower_name.startswith(".env.") and lower_name != ".env.example":
        return "environment override file"
    if lower_name.endswith(".pyc"):
        return "compiled Python file"
    return None


def _record_exclusion(excluded: list[dict], rel: Path, reason: str):
    excluded.append({"path": rel.as_posix(), "reason": reason, "status": "EXCLUDED"})


def build_package_plan(root: Path = PROJECT_ROOT) -> dict:
    """Return the exact allow-listed files that the package builder would copy.

    This function is intentionally shared with build_acceptance_package.py so
    the checker validates the same selection rules as the packager.
    """
    root = Path(root)
    files: list[dict] = []
    excluded: list[dict] = []
    seen: set[str] = set()

    def add(source_rel: Path, destination_rel: Path, category: str):
        source_key = source_rel.as_posix()
        destination_key = destination_rel.as_posix()
        if destination_key in seen:
            return
        reason = _forbidden_reason(source_rel) or _forbidden_reason(destination_rel)
        if reason:
            _record_exclusion(excluded, source_rel, reason)
            return
        source = root / source_rel
        if not source.is_file() or source.is_symlink():
            _record_exclusion(excluded, source_rel, "missing, non-regular, or symlink")
            return
        size = source.stat().st_size
        if size > MAX_PACKAGE_FILE_BYTES:
            _record_exclusion(excluded, source_rel, f"over 50 MiB ({size} bytes)")
            return
        files.append({
            "source": source_key,
            "path": destination_key,
            "category": category,
            "bytes": size,
        })
        seen.add(destination_key)

    for name in CANDIDATE_ROOT_FILES:
        add(Path(name), Path(name), "root")

    # The formal inputs are named explicitly. The 42 result files come from the
    # frozen snapshot rather than from a recursive directory sweep.
    add(Path("data/eval/qa_set.jsonl"), Path("data/eval/qa_set.jsonl"), "data")
    corpus = root / "data" / "corpus"
    if corpus.is_dir():
        for source in sorted(corpus.rglob("*.docx")):
            rel = source.relative_to(root)
            add(rel, rel, "data")

    try:
        snapshot_candidates = [
            root / "delivery" / "formal_results_snapshot.json",
            root / "repair_outputs" / "acceptance_20261023" / "delivery" / "formal_results_snapshot.json",
        ]
        snapshot_path = next(path for path in snapshot_candidates if path.exists())
        snapshot = read_json(snapshot_path)
        for name in sorted(snapshot.get("files", {})):
            rel = Path("results") / name
            add(rel, rel, "results")
    except Exception as exc:
        _record_exclusion(excluded, Path("results"), f"snapshot unavailable: {exc!r}")

    for group, names, category in (
        ("src", SOURCE_FILES, "src"),
        ("experiments", EXPERIMENT_FILES, "experiments"),
        ("demo", DEMO_FILES, "demo"),
        ("scripts", SCRIPT_FILES, "scripts"),
    ):
        for name in names:
            add(Path(name), Path(name), category)

    delivery_candidates = [
        root / "delivery",
        root / "repair_outputs" / "acceptance_20261023" / "delivery",
    ]
    delivery = next((p for p in delivery_candidates if p.is_dir()), delivery_candidates[0])
    duplicate_delivery_names = {
        "报告_精简_edge.pdf",
    }
    if delivery.is_dir():
        for source in sorted(delivery.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(root)
            delivery_rel = source.relative_to(delivery)
            if any(
                part.startswith("_edge_profile")
                for part in delivery_rel.parts
            ):
                _record_exclusion(excluded, rel, "isolated browser profile used for PDF printing")
                continue
            if any(part == "lo_profile" for part in delivery_rel.parts):
                _record_exclusion(excluded, rel, "temporary LibreOffice profile")
                continue
            if source.parent == delivery and source.name in duplicate_delivery_names:
                _record_exclusion(excluded, rel, "duplicate intermediate report render")
                continue
            if source.parent == delivery and source.name in GENERATED_DELIVERY_NAMES:
                _record_exclusion(excluded, rel, "regenerated package manifest/checksum")
                continue
            destination = Path("delivery") / delivery_rel
            add(rel, destination, "delivery")

    root_pdf = (root / "报告.pdf") if (root / "报告.pdf").is_file() else None
    concise_pdf = delivery / "报告_精简.pdf"
    source_rel = None
    destination_rel = None
    if root_pdf is not None:
        source_rel = root_pdf.relative_to(root)
        destination_rel = Path("报告.pdf")
    elif concise_pdf.is_file():
        source_rel = concise_pdf.relative_to(root)
        destination_rel = Path("报告.pdf")
    if source_rel is not None:
        add(source_rel, destination_rel, "root")

    # Record known out-of-scope material without enumerating or reading it.
    for name, reason in (
        (".env", "secret environment file"),
        (".git", "git metadata"),
        (".tmp_docx", "temporary DOCX extraction"),
        ("repair_outputs", "internal repair/audit workspace outside curated inputs"),
    ):
        if (root / name).exists():
            _record_exclusion(excluded, Path(name), reason)

    for rel, reason, status in (
        (
            "repair_outputs/acceptance_20261023/main_new_results_audit.json",
            "internal audit note; curated copies are selected individually",
            "EXCLUDED",
        ),
    ):
        if (root / rel).exists():
            excluded.append({"path": rel, "reason": reason, "status": status})

    # Keep the deliberate exclusions to genuine debug leftovers. The T4 runner and
    # frozen queries are part of the reproducible package rather than scratch files.
    for name in ("scripts/qa_annotation_draft.csv", "scripts/_debug_ms.py"):
        if (root / name).exists():
            _record_exclusion(excluded, Path(name), "debug or uncollected candidate")

    # Reviewed new-result supplements. Formal results/ is not touched; these
    # curated files are mapped into the new submission tree under delivery/provenance.
    supplement_specs = (
        (
            'repair_outputs/acceptance_20261023/t2/fig_t2_21_configs.plot_source.json',
            'results/acceptance_20261023/t2/fig_t2_21_configs.plot_source.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/fig_t2_21_configs.png',
            'results/acceptance_20261023/t2/fig_t2_21_configs.png',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/fig_t2_advanced_chunking.plot_source.json',
            'results/acceptance_20261023/t2/fig_t2_advanced_chunking.plot_source.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/fig_t2_advanced_chunking.png',
            'results/acceptance_20261023/t2/fig_t2_advanced_chunking.png',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/finalize_and_figure.py',
            'results/acceptance_20261023/t2/finalize_and_figure.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/manifest.json',
            'results/acceptance_20261023/t2/manifest.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/pristine_t2_advanced_chunking.csv',
            'results/acceptance_20261023/t2/pristine_t2_advanced_chunking.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/pristine_t2_advanced_chunking_cases.csv',
            'results/acceptance_20261023/t2/pristine_t2_advanced_chunking_cases.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/recovery_provenance.json',
            'results/acceptance_20261023/t2/recovery_provenance.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/retry_errors_only.py',
            'results/acceptance_20261023/t2/retry_errors_only.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_21_configs_analysis.md',
            'results/acceptance_20261023/t2/t2_21_configs_analysis.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking_cases.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking_cases.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking_cases_final.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking_cases_final.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking_cases_with_retry.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking_cases_with_retry.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking_final.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking_final.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t2/t2_advanced_chunking_retry_attempts.csv',
            'results/acceptance_20261023/t2/t2_advanced_chunking_retry_attempts.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/_t3_progress.csv',
            'results/acceptance_20261023/t3/_t3_progress.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/alpha_run.log',
            'results/acceptance_20261023/t3/alpha_run.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/control_answers_run.log',
            'results/acceptance_20261023/t3/control_answers_run.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/control_run.log',
            'results/acceptance_20261023/t3/control_run.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/fig_t3_alpha_acceptance.png',
            'results/acceptance_20261023/t3/fig_t3_alpha_acceptance.png',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/meta.json',
            'results/acceptance_20261023/t3/meta.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/q28_failed_to_success_case.md',
            'results/acceptance_20261023/t3/q28_failed_to_success_case.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run.log',
            'results/acceptance_20261023/t3/run.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_alpha_sweep.py',
            'results/acceptance_20261023/t3/run_alpha_sweep.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_metadata.json',
            'results/acceptance_20261023/t3/run_metadata.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/RUN_PROVENANCE.md',
            'results/acceptance_20261023/t3/RUN_PROVENANCE.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_subquery_parallel.log',
            'results/acceptance_20261023/t3/run_subquery_parallel.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_t3_control.py',
            'results/acceptance_20261023/t3/run_t3_control.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_t3_real.py',
            'results/acceptance_20261023/t3/run_t3_real.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/run_t3_subquery_parallel.py',
            'results/acceptance_20261023/t3/run_t3_subquery_parallel.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_alpha_per_question.csv',
            'results/acceptance_20261023/t3/t3_alpha_per_question.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_cases_baseline.csv',
            'results/acceptance_20261023/t3/t3_cases_baseline.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_cases_multi.csv',
            'results/acceptance_20261023/t3/t3_cases_multi.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_cases_rewrite.csv',
            'results/acceptance_20261023/t3/t3_cases_rewrite.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_cases_subquery.csv',
            'results/acceptance_20261023/t3/t3_cases_subquery.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_linear_alpha_sweep.csv',
            'results/acceptance_20261023/t3/t3_linear_alpha_sweep.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_subquery_checkpoint.jsonl',
            'results/acceptance_20261023/t3/t3_subquery_checkpoint.jsonl',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/t3_summary.csv',
            'results/acceptance_20261023/t3/t3_summary.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t3/test_run_t3_subquery_parallel.py',
            'results/acceptance_20261023/t3/test_run_t3_subquery_parallel.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/build_consolidated_cases.py',
            'results/acceptance_20261023/t4/build_consolidated_cases.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/pilot_1_queries_metrics.jsonl',
            'results/acceptance_20261023/t4/pilot_1_queries_metrics.jsonl',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/pilot_1_raw.json',
            'results/acceptance_20261023/t4/pilot_1_raw.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/pilot_2_queries_metrics.jsonl',
            'results/acceptance_20261023/t4/pilot_2_queries_metrics.jsonl',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/pilot_2_raw.json',
            'results/acceptance_20261023/t4/pilot_2_raw.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/pilot_recovery_manifest.json',
            'results/acceptance_20261023/t4/pilot_recovery_manifest.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/run_t4_acceptance.pre_final.py',
            'results/acceptance_20261023/t4/run_t4_acceptance.pre_final.py',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_acceptance_checker.json',
            'results/acceptance_20261023/t4/t4_acceptance_checker.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_acceptance_manifest.json',
            'results/acceptance_20261023/t4/t4_acceptance_manifest.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_acceptance_per_query.json',
            'results/acceptance_20261023/t4/t4_acceptance_per_query.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_acceptance_top10.jsonl',
            'results/acceptance_20261023/t4/t4_acceptance_top10.jsonl',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_case_10.csv',
            'results/acceptance_20261023/t4/t4_case_10.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_case_10.json',
            'results/acceptance_20261023/t4/t4_case_10.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_case_10.md',
            'results/acceptance_20261023/t4/t4_case_10.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_case_checker.json',
            'results/acceptance_20261023/t4/t4_case_checker.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_consolidated_80.json',
            'results/acceptance_20261023/t4/t4_consolidated_80.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_consolidated_manifest.json',
            'results/acceptance_20261023/t4/t4_consolidated_manifest.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_freeze.json',
            'results/acceptance_20261023/t4/t4_freeze.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_freeze_final.json',
            'results/acceptance_20261023/t4/t4_freeze_final.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t4/t4_run_claim_final.json',
            'results/acceptance_20261023/t4/t4_run_claim_final.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/00_offline_initial.txt',
            'results/acceptance_20261023/t7/00_offline_initial.txt',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/01_cache_validation.txt',
            'results/acceptance_20261023/t7/01_cache_validation.txt',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/02_llm_judge_run.txt',
            'results/acceptance_20261023/t7/02_llm_judge_run.txt',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/03_judge_issues.md',
            'results/acceptance_20261023/t7/03_judge_issues.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/04_worst5_evidence.md',
            'results/acceptance_20261023/t7/04_worst5_evidence.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/04_worst5_provenance.csv',
            'results/acceptance_20261023/t7/04_worst5_provenance.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/04_worst5_provenance.json',
            'results/acceptance_20261023/t7/04_worst5_provenance.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/fig_t7_level.png',
            'results/acceptance_20261023/t7/fig_t7_level.png',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/fig_t7_source_hash.json',
            'results/acceptance_20261023/t7/fig_t7_source_hash.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/judge_checkpoint.jsonl',
            'results/acceptance_20261023/t7/judge_checkpoint.jsonl',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/rebuild_provenance.json',
            'results/acceptance_20261023/t7/rebuild_provenance.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/t7_llm_judge.csv',
            'results/acceptance_20261023/t7/t7_llm_judge.csv',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/t7/t7_llm_judge_summary.json',
            'results/acceptance_20261023/t7/t7_llm_judge_summary.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/check_demo_summary.txt',
            'results/acceptance_20261023/demo/check_demo_summary.txt',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/demo_acceptance.log',
            'results/acceptance_20261023/demo/demo_acceptance.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/demo_script_5min.md',
            'results/acceptance_20261023/demo/demo_script_5min.md',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/http_e2e.log',
            'results/acceptance_20261023/demo/http_e2e.log',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/http_e2e_results.json',
            'results/acceptance_20261023/demo/http_e2e_results.json',
            "results",
        ),
        (
            'repair_outputs/acceptance_20261023/demo/http_server.out.txt',
            'results/acceptance_20261023/demo/http_server.out.txt',
            "results",
        ),
    )

    for src, dst, category in supplement_specs:
        source_rel = Path(src)
        destination_rel = Path(dst)
        if not (root / source_rel).is_file() and (root / destination_rel).is_file():
            # In an unpacked submission tree (without repair_outputs/), the
            # already-curated destination is the only trustworthy source.
            source_rel = destination_rel
        add(source_rel, destination_rel, category)

    # Mirror the reviewed acceptance trees by relative path so any new
    # runner/query/trace file is included with its original file name.
    reviewed_dirs = ("t2", "t3", "t4", "t7", "demo")
    repair_acceptance_root = root / "repair_outputs" / "acceptance_20261023"
    if not repair_acceptance_root.is_dir():
        # In an unpacked submission tree, results/acceptance_20261023 is the
        # curated mirror and can be re-validated / re-packed directly.
        repair_acceptance_root = root / "results" / "acceptance_20261023"
    for group in reviewed_dirs:
        source_dir = repair_acceptance_root / group
        if not source_dir.is_dir():
            continue
        for source in sorted(source_dir.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            rel_parts = source.relative_to(repair_acceptance_root).parts
            if any(
                part in {"__pycache__", "_cache", "_tmp", ".pytest_cache", ".mypy_cache"}
                for part in rel_parts
            ):
                continue
            destination = Path("results") / "acceptance_20261023" / source.relative_to(repair_acceptance_root)
            add(source.relative_to(root), destination, "results")

    for zip_name in ("\u5b66\u53f7_RAG\u4f5c\u4e1a.zip", OUTPUT_NAME):
        old_zip = root.parent / zip_name
        if old_zip.exists():
            excluded.append({
                "path": str(old_zip),
                "reason": "existing zip, never overwritten or packaged",
                "status": "EXCLUDED",
            })

    files.sort(key=lambda item: item["path"])
    excluded.sort(key=lambda item: item["path"])
    return {
        "files": files,
        "excluded": excluded,
        "categories": dict(Counter(item["category"] for item in files)),
        "count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def check_qa_set(checks: list[Check]) -> dict:
    try:
        actual_hash = sha256_file(QA_SET)
        rows = read_jsonl(QA_SET)
        ids = [str(row.get("id", "")).strip() for row in rows]
        levels = Counter(str(row.get("level", "")).strip() for row in rows)
        ok = (
            actual_hash == QA_SHA256
            and len(rows) == 30
            and len(set(ids)) == 30
            and all(ids)
            and dict(levels) == EXPECTED_LEVELS
        )
        detail = (
            f"sha256={actual_hash}, questions={len(rows)}, unique_ids={len(set(ids))}, "
            f"levels={dict(sorted(levels.items()))}"
        )
        _add(checks, "QA_SET", "fixed 30-question set and level counts", ok, detail, hard=True)
        return {"ids": set(ids), "levels": dict(levels), "hash": actual_hash}
    except Exception as exc:
        _add(checks, "QA_SET", "fixed 30-question set and level counts", False, repr(exc), hard=True)
        return {"ids": set(), "levels": {}, "hash": None}


def check_results_snapshot(checks: list[Check]) -> None:
    try:
        snapshot = read_json(SNAPSHOT)
        expected = snapshot.get("files", {})
        # Keep the frozen 42-file formal snapshot scoped to top-level results/
        # files. Reviewed 2026-10-23 supplements live under
        # results/acceptance_20261023/ and are checked by PACKAGE_INVENTORY.
        disk_files = sorted(path for path in RESULTS.glob("*") if path.is_file())
        disk_map = {path.relative_to(RESULTS).as_posix(): path for path in disk_files}
        supplement_files = sorted(path for path in RESULTS.rglob("*") if path.is_file())
        supplement_count = len([p for p in supplement_files if "acceptance_20261023" in p.parts])
        expected_keys = set(expected)
        disk_keys = set(disk_map)

        mismatches = []
        for key in sorted(expected_keys & disk_keys):
            path = disk_map[key]
            actual_sha = sha256_file(path)
            actual_bytes = path.stat().st_size
            want = expected[key]
            if actual_sha != want.get("sha256") or actual_bytes != want.get("bytes"):
                mismatches.append(
                    f"{key}: sha256={actual_sha != want.get('sha256')}, "
                    f"bytes={actual_bytes}/{want.get('bytes')}"
                )

        ok = (
            snapshot.get("file_count") == EXPECTED_RESULT_COUNT
            and len(expected) == EXPECTED_RESULT_COUNT
            and len(disk_files) == EXPECTED_RESULT_COUNT
            and expected_keys == disk_keys
            and not mismatches
        )
        detail = (
            f"snapshot={len(expected)}, disk_top_level={len(disk_files)}, "
            f"missing={sorted(expected_keys - disk_keys)}, "
            f"unexpected={sorted(disk_keys - expected_keys)}, "
            f"mismatches={mismatches[:5]}, supplements={supplement_count}"
        )
        _add(checks, "RESULTS_SNAPSHOT", "42 formal results match snapshot bytes", ok, detail, hard=True)
    except Exception as exc:
        _add(checks, "RESULTS_SNAPSHOT", "42 formal results match snapshot bytes", False, repr(exc), hard=True)


def check_t1(checks: list[Check], qa_ids: set[str]) -> None:
    try:
        rows = read_jsonl(RESULTS / "t1_baseline.jsonl")
        ids = [str(row.get("id", "")).strip() for row in rows]
        ok = len(rows) == 30 and len(set(ids)) == 30 and set(ids) == qa_ids
        _add(
            checks,
            "T1_COUNT",
            "T1 has 30 RAG generations matching the question set",
            ok,
            f"rows={len(rows)}, unique_ids={len(set(ids))}, ids_match={set(ids) == qa_ids}",
        )
    except Exception as exc:
        _add(checks, "T1_COUNT", "T1 has 30 RAG generations matching the question set", False, repr(exc))


def check_t2(checks: list[Check]) -> None:
    try:
        configs = read_csv(RESULTS / "t2_chunking.csv")
        cases = read_csv(RESULTS / "t2_chunking_cases.csv")
        config_keys = {
            (row.get("method", ""), row.get("size", ""), row.get("overlap", ""))
            for row in configs
        }
        case_keys = {
            (row.get("method", ""), row.get("size", ""), row.get("overlap", ""))
            for row in cases
        }
        per_config = Counter(
            (row.get("method", ""), row.get("size", ""), row.get("overlap", ""))
            for row in cases
        )
        errors = sum(int(row.get("n_generation_errors") or 0) for row in configs)
        balanced = len(per_config) == 15 and all(count == 30 for count in per_config.values())
        ok = (
            len(configs) == 15
            and len(config_keys) == 15
            and len(cases) == 450
            and case_keys == config_keys
            and balanced
            and errors == 0
        )
        _add(
            checks,
            "T2_COUNT",
            "T2 has 15 configurations and 450 case rows",
            ok,
            f"configs={len(configs)}, unique_configs={len(config_keys)}, "
            f"cases={len(cases)}, all_30_per_config={balanced}, generation_errors={errors}",
        )
    except Exception as exc:
        _add(checks, "T2_COUNT", "T2 has 15 configurations and 450 case rows", False, repr(exc))


def check_t4(checks: list[Check]) -> dict:
    result = {}
    try:
        rows = read_csv(RESULTS / "t4_retriever.csv")
        by_name = {row.get("retriever", ""): row for row in rows}
        exact = len(rows) == 3 and set(by_name) == set(T4_EXPECTED)
        mismatches = []
        if exact:
            for name, metrics in T4_EXPECTED.items():
                for column, expected in metrics.items():
                    actual = by_name[name].get(column, "")
                    if not _decimal_equal(actual, expected):
                        mismatches.append(f"{name}.{column}={actual} (expected {expected})")
        ok = exact and not mismatches
        _add(
            checks,
            "T4_METRICS",
            "T4 has exactly three rows and the reviewed retrieval metrics",
            ok,
            f"rows={len(rows)}, retrievers={sorted(by_name)}, mismatches={mismatches}",
        )
        result = by_name
    except Exception as exc:
        _add(
            checks,
            "T4_METRICS",
            "T4 has exactly three rows and the reviewed retrieval metrics",
            False,
            repr(exc),
        )
    return result


def check_t5(checks: list[Check]) -> None:
    try:
        rerank = read_csv(RESULTS / "t5_rerank.csv")
        compress = read_csv(RESULTS / "t5_compress.csv")
        variants = Counter(row.get("variant", "") for row in rerank)
        pairs = {(row.get("id", ""), row.get("variant", "")) for row in rerank}
        ok = (
            len(rerank) == 90
            and len(pairs) == 90
            and variants == Counter({"direct": 30, "rules": 30, "cross_encoder": 30})
            and len(compress) == 18
        )
        _add(
            checks,
            "T5_COUNT",
            "T5 has 90 rerank rows and 18 paired compression rows",
            ok,
            f"rerank={len(rerank)}, unique_pairs={len(pairs)}, "
            f"variants={dict(sorted(variants.items()))}, compress={len(compress)}",
        )
    except Exception as exc:
        _add(checks, "T5_COUNT", "T5 has 90 rerank rows and 18 paired compression rows", False, repr(exc))


def check_t6(checks: list[Check]) -> None:
    try:
        rows = read_csv(RESULTS / "t6_retry.csv")
        trace = read_jsonl(RESULTS / "t6_retry_trace.jsonl")
        summary = read_json(RESULTS / "t6_retry_summary.json")
        second = sum(str(row.get("second_retrieval", "")).strip().lower() == "true" for row in rows)
        ok = (
            len(rows) == 6
            and len(trace) == 6
            and second == 4
            and summary.get("sample_size") == 6
            and summary.get("second_retrieval_count") == 4
        )
        _add(
            checks,
            "T6_RETRY",
            "T6 has six retry cases and four second retrievals",
            ok,
            f"csv={len(rows)}, trace={len(trace)}, second_retrieval={second}, "
            f"summary_sample={summary.get('sample_size')}, "
            f"summary_second={summary.get('second_retrieval_count')}",
        )
    except Exception as exc:
        _add(checks, "T6_RETRY", "T6 has six retry cases and four second retrievals", False, repr(exc))


def check_t1_evidence(checks: list[Check]) -> None:
    try:
        path = DELIVERY / "evidence" / "t1_case_evidence.json"
        evidence = read_json(path)
        cases = evidence.get("cases", [])
        by_id = {str(case.get("id", "")): case for case in cases}
        present = REQUIRED_T1_EVIDENCE_IDS <= set(by_id)
        details = []
        ok = present and len(cases) == 4 and len(by_id) == 4
        for case_id in sorted(REQUIRED_T1_EVIDENCE_IDS):
            case = by_id.get(case_id)
            if case is None:
                details.append(f"{case_id}: missing")
                ok = False
                continue
            semantic = case.get("direct_semantic_answered")
            heuristic = "direct_answered_heuristic" in case
            legacy = str(case.get("legacy_judge_scope", ""))
            case_ok = semantic is False and heuristic and "legacy reference only" in legacy
            details.append(
                f"{case_id}: semantic={semantic!r}, heuristic_present={heuristic}, "
                f"legacy_scope_ok={'legacy reference only' in legacy}"
            )
            ok = ok and case_ok
        _add(
            checks,
            "T1_EVIDENCE",
            "four T1 cases preserve semantic direct-answer distinctions",
            ok,
            "; ".join(details),
        )
    except Exception as exc:
        _add(
            checks,
            "T1_EVIDENCE",
            "four T1 cases preserve semantic direct-answer distinctions",
            False,
            repr(exc),
        )


def check_lock(checks: list[Check]) -> None:
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except Exception as exc:
        _add(
            checks,
            "REQUIREMENTS_LOCK",
            "requirements lock is parseable with no missing placeholders",
            False,
            f"packaging import failed: {exc!r}",
            hard=True,
        )
        return

    try:
        lines = LOCK_FILE.read_text(encoding="utf-8").splitlines()
        requirements = [
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ]
        parsed = []
        errors = []
        placeholders = []
        for line in requirements:
            if re.search(r"(?i)==\s*<?missing>?\s*$", line.split("#", 1)[0].strip()):
                placeholders.append(line)
            try:
                requirement = Requirement(line)
                parsed.append(requirement)
                if requirement.name.upper() == "MISSING":
                    placeholders.append(line)
            except InvalidRequirement as exc:
                errors.append(f"{line}: {exc}")
        status = read_json(LOCK_STATUS)
        ok = bool(requirements) and not errors and not placeholders and status.get("status") == "PARTIAL"
        _add(
            checks,
            "REQUIREMENTS_LOCK",
            "requirements lock is parseable with no missing placeholders",
            ok,
            f"requirements={len(requirements)}, parsed={len(parsed)}, errors={errors}, "
            f"placeholders={placeholders}, status={status.get('status')!r}",
            hard=bool(errors or placeholders or not requirements),
        )
    except Exception as exc:
        _add(
            checks,
            "REQUIREMENTS_LOCK",
            "requirements lock is parseable with no missing placeholders",
            False,
            repr(exc),
            hard=True,
        )


def check_exclusions(checks: list[Check], plan: dict) -> None:
    bad = [item["path"] for item in plan["files"] if _forbidden_reason(Path(item["path"]))]
    oversize = [item["path"] for item in plan["files"] if item["bytes"] > MAX_PACKAGE_FILE_BYTES]
    delivery_bad = []
    if DELIVERY.is_dir():
        for path in DELIVERY.rglob("*"):
            if path.is_file():
                rel = path.relative_to(DELIVERY)
                if _forbidden_reason(rel):
                    delivery_bad.append(rel.as_posix())
    ok = not bad and not oversize and not delivery_bad
    _add(
        checks,
        "PACKAGE_EXCLUSIONS",
        "delivery and package candidates exclude secrets, VCS, caches, bytecode, and oversize files",
        ok,
        f"forbidden={bad[:10]}, oversize={oversize[:10]}, delivery_forbidden={delivery_bad[:10]}, "
        f"planned_files={plan['count']}",
    )


def check_package_inventory(checks: list[Check], plan: dict) -> None:
    corpus = [
        item for item in plan["files"]
        if item["path"].startswith("data/corpus/") and item["path"].lower().endswith(".docx")
    ]
    # Formal results are the frozen top-level 42 files. Reviewed supplements live
    # under results/acceptance_20261023/{t2,t3,t4,t7,demo} and are required too,
    # but they must not be miscounted as the frozen 42-file snapshot.
    result_files = [
        item for item in plan["files"]
        if item["path"].startswith("results/")
        and not item["path"].startswith("results/acceptance_20261023/")
    ]
    accepted_subdirs = {
        "t2": "results/acceptance_20261023/t2",
        "t3": "results/acceptance_20261023/t3",
        "t4": "results/acceptance_20261023/t4",
        "t7": "results/acceptance_20261023/t7",
        "demo": "results/acceptance_20261023/demo",
    }
    selected = {item["path"] for item in plan["files"]}
    supplement_summary = {
        name: len([p for p in selected if p.startswith(prefix + "/")])
        for name, prefix in accepted_subdirs.items()
    }
    required_paths = {
        "README.md",
        "\u62a5\u544a.md",
        "\u62a5\u544a.pdf",
        "requirements.txt",
        ".env.example",
        "data/eval/qa_set.jsonl",
        "delivery/requirements-lock.txt",
        "delivery/requirements-lock-status.json",
        "delivery/formal_results_snapshot.json",
    }
    selected = {item["path"] for item in plan["files"]}
    missing = sorted(required_paths - selected)
    ok = (
        len(corpus) == 43
        and len(result_files) == 42
        and all(supplement_summary.values())
        and not missing
    )
    _add(
        checks,
        "PACKAGE_INVENTORY",
        "package plan contains 43 corpus DOCX files, all 42 results, and required root files",
        ok,
        f"corpus_docx={len(corpus)}, formal_results={len(result_files)}, "
        f"acceptance_supplements={supplement_summary}, missing={missing}",
    )


def required_human_items(plan: dict) -> list[dict]:
    items = [
        {
            "id": "STUDENT_ID",
            "status": "REQUIRED_HUMAN",
            "detail": "Real student IDs and the replacement submission filename are not available in the repository; no placeholder may be presented as accepted.",
        },
        {
            "id": "TEAM_DIVISION",
            "status": "REQUIRED_HUMAN",
            "detail": "No trustworthy member names or division of work exist; do not reconstruct or invent them.",
        },
    ]

    items.append({
        "id": "GOLD_HUMAN_REVIEW",
        "status": "REQUIRED_HUMAN",
        "detail": "T3 subquery/control/alpha CSV and audit are complete; whether a human reviewed the gold answers cannot be proven from the repository and must be confirmed by the user, not auto-attested by AI.",
    })
    items.append({
        "id": "T2_SUPPLEMENT_INCLUDED",
        "status": "INCLUDED",
        "detail": "T2 final and advanced CSVs are included as supplemental evidence; this inclusion does not mark T3/T4/T7/Demo complete.",
    })
    items.extend([
        {
            "id": "T3_UNRESOLVED",
            "status": "UNRESOLVED",
            "detail": "T3 strategy/control/alpha artifacts exist, but no real-model T3 rerun or human review was completed in this thread. Do not claim T3 is complete.",
        },
        {
            "id": "T4_UNRESOLVED",
            "status": "UNRESOLVED",
            "detail": "T4 formal metrics and selected-case artifacts exist, but the dense-winner target was not established as a complete experiment; D4 remains PARTIAL.",
        },
        {
            "id": "T7_UNRESOLVED",
            "status": "UNRESOLVED",
            "detail": "T7 uses a same-model LLM judge (not an independent judge or human truth). Treat the metrics as transparent reference, not completed acceptance evidence.",
        },
        {
            "id": "DEMO_UNRESOLVED",
            "status": "UNRESOLVED",
            "detail": "HTTP queue L1/L4 candidate records and stub tests exist, but browser visual verification and recording were not completed.",
        },
    ])
    items.append({
        "id": "D5_REPORT_LAYOUT",
        "status": "UNRESOLVED",
        "detail": "The 8-page PDF is structurally valid and extractable, but printed pages still show host file:/// URL footers; layout remains a visible defect.",
    })
    items.append({
        "id": "PACKAGE_PLAN",
        "status": "NOT_ACCEPTED",
        "detail": f"Technical package plan contains {plan['count']} files; global status remains NOT_ACCEPTED.",
    })
    return items


def run_checks(write_report: bool = True) -> dict:
    checks: list[Check] = []
    qa = check_qa_set(checks)
    check_results_snapshot(checks)
    check_t1(checks, qa["ids"])
    check_t2(checks)
    check_t4(checks)
    check_t5(checks)
    check_t6(checks)
    check_t1_evidence(checks)
    check_lock(checks)
    plan = build_package_plan(PROJECT_ROOT)
    check_exclusions(checks, plan)
    check_package_inventory(checks, plan)

    failures = [check for check in checks if not check.ok]
    hard_failures = [check for check in failures if check.hard]
    pass_count = sum(check.ok for check in checks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "global_status": "NOT_ACCEPTED",
        "technical_status": "FAIL" if failures else "PASS",
        "hard_fail": bool(hard_failures),
        "summary": {
            "pass": pass_count,
            "fail": len(failures),
            "required_human": sum(
                1 for item in required_human_items(plan)
                if item["status"] == "REQUIRED_HUMAN"
            ),
        },
        "checks": [check.as_dict() for check in checks],
        "required_human": required_human_items(plan),
        "package_plan": {
            "file_count": plan["count"],
            "total_bytes": plan["total_bytes"],
            "categories": plan["categories"],
            "excluded_count": len(plan["excluded"]),
            "excluded_examples": plan["excluded"][:20],
        },
        "new_results_policy": {
            "formal_snapshot_unchanged": True,
            "formal_snapshot_file_count": EXPECTED_RESULT_COUNT,
            "included_new_supplements": [
                "repair_outputs/acceptance_20261023/t2",
                "repair_outputs/acceptance_20261023/t3",
                "repair_outputs/acceptance_20261023/t4",
                "repair_outputs/acceptance_20261023/t7",
                "repair_outputs/acceptance_20261023/demo",
            ],
            "unresolved": ["T3", "T4", "T7", "Demo"],
            "status": "PARTIAL",
            "detail": "Supplemental T2/T3/T4/T7/Demo artifacts are included without changing the frozen formal results/ 42 files. T3/T4/T7/Demo remain unresolved experimental or human-review items.",
        },
    }
    if write_report:
        CHECK_REPORT.parent.mkdir(parents=True, exist_ok=True)
        CHECK_REPORT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return report


def print_report(report: dict) -> None:
    print("=== Submission Acceptance Check ===")
    for check in report["checks"]:
        print(f"{check['status']:<6} [{check['id']}] {check['detail']}")
    for item in report["required_human"]:
        print(f"{item['status']:<14} [{item['id']}] {item['detail']}")
    print(
        "Summary: "
        f"PASS={report['summary']['pass']} "
        f"FAIL={report['summary']['fail']} "
        f"REQUIRED_HUMAN={report['summary']['required_human']}"
    )
    print(f"Technical status: {report['technical_status']}")
    print("Global status: NOT_ACCEPTED")
    print(f"Report: {CHECK_REPORT}")


def _print_full_plan() -> None:
    print("FULL MODE: no costly work is run by this checker.")
    print("Only the user may opt in to the following expensive experiments:")
    print("  - T3 new subquery branch: requires a configured LLM service and network access.")
    print("  - T7 independent judge: requires a separately credentialed model or human scoring.")
    print("  - Demo end-to-end: requires starting the local Gradio app and manual visual review.")
    print("  - T4 dense winner supplement: requires an approved retrieval rerun and human acceptance.")
    print("No dependency install, API call, service start, or formal-results write is performed.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="offline read-only checks only (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="print the costly experiments the user would need to run; do not run them",
    )
    args = parser.parse_args(argv)

    report = run_checks(write_report=True)
    print_report(report)
    if args.full:
        _print_full_plan()
    return 1 if report["technical_status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
