"""Build the reviewed RAG submission package without overwriting history.

Default mode is a dry run:

  python -I -B scripts/build_acceptance_package.py --dry-run

Apply mode first reruns the fail-closed checker, copies only allow-listed files
to an isolated staging directory, generates delivery/manifest.json and
SHA256SUMS.txt there, and writes a new zip with exclusive-create semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import check_submission_acceptance as acceptance  # noqa: E402

OUTPUT_NAME = acceptance.OUTPUT_NAME
DEFAULT_OUTPUT = PROJECT_ROOT.parent / OUTPUT_NAME
HARD_FAIL_IDS = {"QA_SET", "RESULTS_SNAPSHOT", "REQUIREMENTS_LOCK"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_plan(plan: dict, output: Path) -> None:
    print("=== Acceptance Package Plan ===")
    print(f"Mode: DRY-RUN (no zip and no staging directory will be written)")
    print(f"Output: {output}")
    print(f"Files: {plan['count']}")
    print(f"Total bytes: {plan['total_bytes']}")
    print(f"Categories: {json.dumps(plan['categories'], ensure_ascii=False, sort_keys=True)}")
    print("Categories breakdown:")
    for name, count in sorted(plan["categories"].items()):
        print(f"  {name}: {count}")
    print(f"Excluded records: {len(plan['excluded'])}")
    grouped = {}
    for item in plan["excluded"]:
        grouped.setdefault(item["reason"], []).append(item)
    for reason, items in sorted(grouped.items()):
        print(f"  EXCLUDED {len(items)} item(s) - {reason}")
        for item in items[:4]:
            print(f"    {item['path']}")
        if len(items) > 4:
            print(f"    ... {len(items) - 4} more")
    print("Required human / not accepted:")
    for item in acceptance.required_human_items(plan):
        print(f"  {item['status']} {item['id']} - {item['detail']}")
    print("Manifest status: NOT_ACCEPTED")


def _check_target(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"output parent does not exist: {output.parent}")
    old_zip = PROJECT_ROOT.parent / "\u5b66\u53f7_RAG\u4f5c\u4e1a.zip"
    if output.resolve() == old_zip.resolve():
        raise FileExistsError("refusing to overwrite the historical 学号_RAG作业.zip")


def _category_for(rel: str, plan_by_path: dict) -> str:
    return plan_by_path.get(rel, {}).get("category", "delivery")


def _pdf_evidence(stage: Path) -> dict:
    pdf = stage / "delivery" / "\u62a5\u544a_\u7cbe\u7b80.pdf"
    evidence = stage / "delivery" / "tests" / "pdf_result.txt"
    result = {
        "paths": [
            "delivery/报告_精简.pdf",
            "报告.pdf",
        ],
        "present": pdf.is_file(),
        "status": "PENDING",
        "evidence": "delivery/tests/pdf_result.txt",
    }
    if not pdf.is_file() or not evidence.is_file():
        # Do not force a pending report PDF to remain unstamped because an
        # ephemeral verification text has not been written yet. The package
        # manifest can still record pages directly from the PDF when available.
        pages = None
        chars = None
        try:
            import fitz
            doc = fitz.open(pdf)
            pages = doc.page_count
            chars = sum(len(page.get_text()) for page in doc)
            doc.close()
        except Exception:
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(pdf))
                pages = len(reader.pages)
                chars = sum(len(page.extract_text() or "") for page in reader.pages)
            except Exception:
                prefix_pages = list((stage / "delivery" / "tests").glob("final_pdf_page*.png"))
                if prefix_pages:
                    nums = []
                    import re as _re
                    for p in prefix_pages:
                        m = _re.search(r"final_pdf_page(\d+)\\.png", p.name)
                        if m:
                            nums.append(int(m.group(1)))
                    if nums:
                        pages = max(nums)
        if pages is not None:
            result["bytes"] = pdf.stat().st_size
            result["pages"] = pages
            result["extractable_text_chars"] = chars
            result["status"] = "PASS" if 1 <= pages <= 8 and chars and chars > 0 else "FAIL"
        return result
    text = evidence.read_text(encoding="utf-8", errors="replace")
    pages_match = re.search(r"(?m)^\s*Page count:\s*(\d+)\s*$", text)
    # Sartre's final render record uses the new phrase:
    #   Extractable text characters (stripped per page): 6788
    # Older render records used:
    #   Extractable text: 7411 characters
    chars_match = re.search(
        r"(?m)^\s*Extractable text characters \(stripped per page\):\s*(\d+)",
        text,
    ) or re.search(
        r"(?m)^\s*Extractable text:\s*(\d+)\s*characters",
        text,
    )
    pages = int(pages_match.group(1)) if pages_match else None
    chars = int(chars_match.group(1)) if chars_match else None
    result.update({
        "paths": result["paths"],
        "bytes": pdf.stat().st_size,
        "pages": pages,
        "extractable_text_chars": chars,
        "status": (
            "PASS" if pages is not None and 1 <= pages <= 8 and chars and chars > 0
            else "FAIL"
        ),
    })
    return result


def _write_manifest(stage: Path, output: Path, plan: dict, check_report: dict) -> dict:
    # The package root is a cleaned submission tree where delivery/ is mapped
    # from the internal acceptance workspace, not nested under repair_outputs/.
    delivery = stage / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    plan_by_path = {item["path"]: item for item in plan["files"]}
    rows = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        if rel in {"delivery/manifest.json", "delivery/SHA256SUMS.txt"}:
            continue
        rows.append({
            "path": rel,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "category": _category_for(rel, plan_by_path),
            "status": "INCLUDED",
        })

    required_human = check_report["required_human"]
    excluded = plan["excluded"]
    manifest = {
        "global_status": "NOT_ACCEPTED",
        "technical_status": check_report["technical_status"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_name": output.name,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "files": rows,
        "unresolved": [
            item for item in required_human
            if item["status"] in {"REQUIRED_HUMAN", "UNRESOLVED"}
        ],
        "excluded": excluded,
        "checks": {
            "technical_status": check_report["technical_status"],
            "pass": check_report["summary"]["pass"],
            "fail": check_report["summary"]["fail"],
            "hard_fail": check_report["hard_fail"],
            "report": "delivery/tests/check_report.json",
        },
        "new_results_policy": check_report["new_results_policy"],
        "artifact_status": {
            "report_pdf": _pdf_evidence(stage),
        },
        "hash_appendix": {
            "data/eval/qa_set.jsonl": {
                "sha256": acceptance.QA_SHA256,
                "bytes": (stage / "data" / "eval" / "qa_set.jsonl").stat().st_size,
            },
            "results/t1_baseline.jsonl": {
                "sha256": _sha256(stage / "results" / "t1_baseline.jsonl"),
                "bytes": (stage / "results" / "t1_baseline.jsonl").stat().st_size,
            },
        },
        "supplemental_artifacts": {
            "t2": "results/acceptance_20261023/t2/t2_advanced_chunking_final.csv",
            "t3": "results/acceptance_20261023/t3/t3_cases_subquery.csv",
            "t4": "results/acceptance_20261023/t4/t4_case_10.json",
            "t7": "results/acceptance_20261023/t7/t7_llm_judge_summary.json",
            "demo": "results/acceptance_20261023/demo/http_e2e_results.json",
        },
        "unresolved_work": check_report["new_results_policy"]["unresolved"],
        "notes": [
            "Global status remains NOT_ACCEPTED; technical verification is not course acceptance.",
            "Excluded internal repair, cache, secret, VCS, temporary, and uncollected candidate files are never copied.",
            "Supplemental T2/T3/T4/T7/Demo artifacts are included under results/acceptance_20261023/{t2,t3,t4,t7,demo} while the frozen top-level results/ 42 files stay unchanged.",
            "T3, T4, T7, and Demo remain unresolved; inclusion is not an acceptance claim.",
        ],
    }
    manifest_path = delivery / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_checksums(stage: Path) -> int:
    rows = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage).as_posix()
        if rel == "delivery/SHA256SUMS.txt":
            continue
        rows.append(f"{_sha256(path)}  {rel}")
    checksum_path = stage / "delivery" / "SHA256SUMS.txt"
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def _zip_stage(stage: Path, output: Path) -> None:
    """Create the zip with exclusive target creation and no history overwrite."""
    temp_zip: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="rag_submission_", suffix=".zip", dir=output.parent, delete=False
        ) as handle:
            temp_zip = Path(handle.name)
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    rel = path.relative_to(stage).as_posix()
                    archive.write(path, rel)
        # x mode is required: an existing destination is never overwritten.
        # If the target appears between _check_target() and here, x mode will
        # fail with FileExistsError and we must leave the existing file intact.
        with temp_zip.open("rb") as source, output.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    except FileExistsError as exc:
        raise RuntimeError(
            f"refusing to replace existing output; destination left intact: {output}"
        ) from exc
    except Exception as exc:
        # Do not unlink output even for a partially copied file: the file may
        # have been created by another process or may be a user artifact. Only
        # our temporary zip is ours to clean up.
        raise RuntimeError(
            f"package build failed; output was NOT deleted. temp_zip cleaned: {temp_zip}. "
            f"output if created is left in place for inspection. error={exc}"
        ) from exc
    finally:
        if temp_zip is not None:
            try:
                temp_zip.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                print(f"WARNING: could not remove temporary zip {temp_zip}")


def apply_package(output: Path) -> dict:
    _check_target(output)
    check_report = acceptance.run_checks(write_report=True)
    failed = [item for item in check_report["checks"] if item["status"] == "FAIL"]
    hard_failed = [item["id"] for item in failed if item["id"] in HARD_FAIL_IDS]
    if hard_failed or failed:
        reasons = ", ".join(item["id"] for item in failed)
        raise RuntimeError(f"refusing to package after technical check failure: {reasons}")

    plan = acceptance.build_package_plan(PROJECT_ROOT)
    with tempfile.TemporaryDirectory(prefix="rag_acceptance_stage_") as temp:
        stage = Path(temp)
        for item in plan["files"]:
            source = Path(item["source"])
            destination = stage / Path(item["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(PROJECT_ROOT / source, destination)

        copied_corpus = sum(
            1 for path in (stage / "data" / "corpus").rglob("*.docx")
            if path.is_file()
        )
        # The frozen 42-file formal results snapshot is exactly the top-level
        # results/* files. reviewed supplements live in
        # results/acceptance_20261023/ and must not be counted here.
        copied_results = sum(1 for path in (stage / "results").glob("*") if path.is_file())
        supplement_dir = stage / "results" / "acceptance_20261023"
        copied_supplements = (
            sum(1 for path in supplement_dir.rglob("*") if path.is_file())
            if supplement_dir.is_dir()
            else 0
        )
        if copied_corpus != 43 or copied_results != 42:
            raise RuntimeError(
                f"staging inventory changed: corpus={copied_corpus}, results={copied_results}"
            )
        if copied_supplements < 1:
            raise RuntimeError(
                f"staging inventory missing reviewed supplements: supplements={copied_supplements}"
            )

        manifest = _write_manifest(stage, output, plan, check_report)
        checksum_count = _write_checksums(stage)
        _zip_stage(stage, output)

    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "file_count": manifest["file_count"] + 2,
        "manifest_file_count": manifest["file_count"],
        "checksum_entries": checksum_count,
        "global_status": "NOT_ACCEPTED",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan only (default)")
    mode.add_argument("--apply", action="store_true", help="create the new zip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new zip path")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    check_report = acceptance.run_checks(write_report=True)
    print("=== Acceptance Package Preflight ===")
    print(f"Technical status: {check_report['technical_status']}")
    print(
        f"Checks: PASS={check_report['summary']['pass']} "
        f"FAIL={check_report['summary']['fail']} "
        f"REQUIRED_HUMAN={check_report['summary']['required_human']}"
    )
    failed = [item for item in check_report["checks"] if item["status"] == "FAIL"]
    hard_failed = [item["id"] for item in failed if item["id"] in HARD_FAIL_IDS]
    if failed:
        for item in failed:
            print(f"FAIL [{item['id']}] {item['detail']}")
    if hard_failed:
        print(f"Fail-closed checks failed: {', '.join(hard_failed)}")
        return 2
    if failed:
        print("Refusing to package while any technical check fails.")
        return 1

    plan = acceptance.build_package_plan(PROJECT_ROOT)
    _print_plan(plan, output)
    if not args.apply:
        print("Dry run complete. No zip was written.")
        return 0

    try:
        result = apply_package(output)
    except Exception as exc:
        print(f"PACKAGE FAILED: {exc}")
        return 3

    print("=== Acceptance Package Result ===")
    print(f"Zip: {result['output']}")
    print(f"Zip bytes: {result['bytes']}")
    print(f"Zip entries: {result['file_count']}")
    print(f"Manifest files: {result['manifest_file_count']}")
    print(f"Checksum entries: {result['checksum_entries']}")
    print(f"Manifest status: {result['global_status']}")
    print("Course acceptance: NOT_ACCEPTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
